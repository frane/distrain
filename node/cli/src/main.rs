#![recursion_limit = "256"]
//! Distrain Node Client — trains and pushes model deltas.
//!
//! Usage:
//!     distrain-node start --config node.toml
//!     distrain-node bootstrap --config node.toml --preset tiny
//!     distrain-node benchmark --config node.toml
//!     distrain-node status --config node.toml

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use distrain_node::{client, data, resources, trainer};
use distrain_shared::config::NodeConfig;
use distrain_shared::storage::Storage;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "distrain-node", about = "Distrain GPU node client")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the training loop
    Start {
        #[arg(long, default_value = "node.toml")]
        config: String,
        /// Force CPU backend (ndarray) instead of GPU
        #[arg(long)]
        cpu: bool,
    },
    /// Create and upload a v0 checkpoint to R2
    Bootstrap {
        #[arg(long, default_value = "node.toml")]
        config: String,
        /// Model preset: micro-test, tiny (125M), small (1B), medium (7B), large (13B)
        #[arg(long, default_value = "tiny")]
        preset: String,
    },
    /// Benchmark this device
    Benchmark {
        #[arg(long, default_value = "node.toml")]
        config: String,
    },
    /// Check training status
    Status {
        #[arg(long, default_value = "node.toml")]
        config: String,
    },
    /// Show effective training parameters
    Config {
        #[arg(long, default_value = "node.toml")]
        config: String,
    },
    /// Evaluate checkpoint(s) — compute loss on real data without training
    Eval {
        #[arg(long, default_value = "node.toml")]
        config: String,
        /// Checkpoint files to evaluate (e.g. v0.safetensors v25.safetensors)
        #[arg(required = true)]
        checkpoints: Vec<String>,
        /// Number of batches to evaluate (default 50)
        #[arg(long, default_value = "50")]
        batches: u64,
    },
    /// Internal: GPU stress test in subprocess (not for direct use)
    #[command(hide = true)]
    GpuStressTest {
        /// Path to checkpoint file
        checkpoint: String,
        #[arg(long)]
        batch_size: usize,
        #[arg(long)]
        seq_len: usize,
    },
    /// Generate text from a checkpoint
    Generate {
        /// Path to checkpoint file
        checkpoint: String,
        /// Prompt text (if empty, generates unconditionally)
        #[arg(long, default_value = "The")]
        prompt: String,
        /// Number of tokens to generate
        #[arg(long, default_value = "128")]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0.8")]
        temperature: f64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Start { config, cpu } => {
            let mut config = load_config(&config)?;
            config.force_cpu = cpu;
            run_training_loop(config).await?;
        }
        Commands::Bootstrap { config, preset } => {
            let config = load_config(&config)?;
            run_bootstrap(&config, &preset).await?;
        }
        Commands::Benchmark { config } => {
            let config = load_config(&config)?;
            run_benchmark(&config).await?;
        }
        Commands::Status { config } => {
            let config = load_config(&config)?;
            show_status(&config).await?;
        }
        Commands::Config { config } => {
            let config = load_config(&config)?;
            show_config(&config).await?;
        }
        Commands::Eval {
            config,
            checkpoints,
            batches,
        } => {
            let config = load_config(&config)?;
            run_eval(&config, &checkpoints, batches).await?;
        }
        Commands::GpuStressTest {
            checkpoint,
            batch_size,
            seq_len,
        } => {
            run_gpu_stress_test_child(&checkpoint, batch_size, seq_len).await?;
        }
        Commands::Generate {
            checkpoint,
            prompt,
            max_tokens,
            temperature,
        } => {
            run_generate(&checkpoint, &prompt, max_tokens, temperature).await?;
        }
    }

    Ok(())
}

fn load_config(path: &str) -> Result<NodeConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config from {path}"))?;
    toml::from_str(&text).context("Failed to parse node config TOML")
}

async fn run_training_loop(mut config: NodeConfig) -> Result<()> {
    let coordinator = client::CoordinatorClient::new(&config.coordinator_url);
    let storage = Storage::new(&config.storage).await?;

    // Load or generate persistent node ID
    let cache_dir_expanded = shellexpand::tilde(&config.cache_dir).to_string();
    let node_id_path = std::path::PathBuf::from(&cache_dir_expanded).join("node_id");
    let persistent_id = if node_id_path.exists() {
        let id = std::fs::read_to_string(&node_id_path)?.trim().to_string();
        info!("Loaded persistent node ID: {id}");
        Some(id)
    } else {
        None
    };

    // Register
    let reg = coordinator.register(&config, persistent_id).await?;
    info!("Registered as {} (api_key starts with {}...)", reg.node_id, &reg.api_key[..8]);

    // Persist the node ID for reuse across restarts
    tokio::fs::create_dir_all(&cache_dir_expanded).await?;
    tokio::fs::write(&node_id_path, &reg.node_id.0).await?;

    // Merge coordinator training params (coordinator is source of truth)
    if let Some(params) = reg.training_params {
        info!(
            "Training params from coordinator: lr={:.2e}→{:.2e}, warmup={:.0}%, grad_clip={}, shards={:.0}%",
            params.lr_max, params.lr_min, params.warmup_fraction * 100.0,
            params.grad_clip_norm, params.shards_fraction * 100.0,
        );
        config.training_params = Some(params);
    }

    let node_id = reg.node_id.0;
    let mut seq_num: u64 = 0;

    // Phase 1: Just check if a GPU adapter exists. No model loading, no calibration.
    // Actual backend selection + H_mini computed from real model in Phase 2.
    let mut h_mini: u64 = config.min_inner_steps; // placeholder, overridden in Phase 2
    if !config.force_cpu {
        let verdict = trainer::probe_gpu().await;
        match verdict {
            trainer::GpuVerdict::Available { name, .. } => {
                info!("GPU adapter found: {name} — will benchmark in Phase 2");
            }
            trainer::GpuVerdict::NoAdapter => {
                info!("No GPU adapter — will use CPU");
                config.force_cpu = true;
            }
        }
    }

    // Cache dir
    let cache_dir = shellexpand::tilde(&config.cache_dir).to_string();
    let cache_dir = std::path::PathBuf::from(cache_dir);
    tokio::fs::create_dir_all(&cache_dir).await?;

    // Load data manifest from R2 (shard list — actual data loaded per round)
    info!("Loading data manifest from storage...");
    let (manifest, data_cache) = data::DataLoader::load_manifest(&storage, &cache_dir)
        .await
        .context("Failed to load data manifest. Run: python tools/prepare_data.py fineweb-edu-10bt --output-dir data/ --upload")?;

    let total_shards = manifest.shards.len();
    let params = config
        .training_params
        .as_ref()
        .cloned()
        .unwrap_or_default();
    let requested_shards = params.shards_per_node(total_shards);
    info!("Data manifest: {total_shards} shards, requested {requested_shards} per round");

    // Will be clamped by memory budget once we know the model size (first checkpoint)
    let mut shards_per_node = requested_shards;
    let mut budget_computed = false;
    // Measured secs/step for the real model on GPU — set by stress test, used by watchdog.
    let mut gpu_secs_per_step: Option<f64> = None;
    // Whether the GPU stress test has been run (only once, on first iteration).
    let mut stress_tested = false;
    // Whether the last round was aborted due to memory pressure.
    let mut memory_pressure_abort = false;
    // Persistent error buffer for compression error feedback across rounds.
    let mut error_buffer = distrain_model::compression::ErrorBuffer::new();
    // Last round's elapsed time (for poll delay estimation).
    let mut result_elapsed: f64 = 0.0;
    // Auto-calibrated batch size and gradient accumulation steps.
    // batch_size = micro_batch_size * grad_accum_steps = effective_batch_size (constant).
    let effective_batch_size = config.batch_size; // target from config (default 4)
    let mut batch_size = effective_batch_size;
    let mut grad_accum_steps: usize = 1;
    let mut batch_calibrated = false;
    // Track checkpoint version to clear error buffer on version change
    let mut last_trained_version: Option<u64> = None;

    loop {
        // Dynamic shard reduction: if previous round hit memory pressure, halve shards
        if memory_pressure_abort {
            let prev = shards_per_node;
            shards_per_node = (shards_per_node / 2).max(1);
            warn!(
                "Memory pressure detected last round — reducing shards from {prev} to {shards_per_node}"
            );
            memory_pressure_abort = false;
        }

        // Get latest checkpoint version
        let ckpt_info = coordinator.get_latest_checkpoint().await?;
        let version = ckpt_info.version;
        info!("Latest checkpoint: v{version}");

        // Clear error buffer when checkpoint version changes — residuals are relative
        // to old weights and no longer valid for the new checkpoint.
        if let Some(prev_v) = last_trained_version {
            if version != prev_v {
                info!("Checkpoint advanced v{prev_v} → v{version} — clearing error buffer");
                error_buffer = distrain_model::compression::ErrorBuffer::new();
            }
        }

        // Download checkpoint if not cached (retry on network errors)
        let ckpt_path = cache_dir.join(format!("v{version}_model.safetensors"));
        if !ckpt_path.exists() {
            let mut download_ok = false;
            for attempt in 1..=3 {
                info!("Downloading checkpoint v{version} (attempt {attempt}/3)...");
                match storage
                    .download_to_file(&ckpt_info.checkpoint_key, &ckpt_path)
                    .await
                {
                    Ok(_) => {
                        download_ok = true;
                        break;
                    }
                    Err(e) => {
                        warn!("Download failed: {e:#}");
                        let _ = tokio::fs::remove_file(&ckpt_path).await;
                        if attempt < 3 {
                            info!("Retrying in 5s...");
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        }
                    }
                }
            }
            if !download_ok {
                warn!("Failed to download checkpoint v{version} after 3 attempts, retrying loop");
                continue;
            }
        }

        // Compute memory budget on first iteration (model size is now known)
        if !budget_computed {
            let model_config = trainer::infer_model_config(&ckpt_path)?;
            match resources::compute_memory_budget(&model_config, &config) {
                Ok(budget) => {
                    if budget.max_shards < shards_per_node {
                        info!(
                            "Clamping shards from {} to {} to fit memory budget ({})",
                            shards_per_node, budget.max_shards, budget,
                        );
                        shards_per_node = budget.max_shards;
                    }
                }
                Err(e) => {
                    // Model doesn't fit in memory at all — fatal
                    return Err(e.context("Memory budget check failed"));
                }
            }
            budget_computed = true;
        }

        // Phase 2: Decide backend and compute H_mini from real model timing.
        //
        // Decision logic (no GPU model loading until we're sure it's safe):
        // 1. If force_cpu or no GPU adapter → CPU
        // 2. If integrated GPU with small buffer → CPU
        // 3. If discrete GPU but max_buffer_size < model size → CPU
        // 4. Otherwise → GPU
        //
        // Then calibrate batch_size + benchmark the chosen backend for H_mini.
        if !stress_tested {
            let target_interval = config.target_push_interval_secs;
            let min_h = config.min_inner_steps;
            let max_h = config.max_inner_steps;
            let model_config = trainer::infer_model_config(&ckpt_path)?;
            // Model weights in BF16 (2 bytes/param) — this is what the GPU actually loads
            let model_weight_bytes = model_config.param_count() as u64 * 2;

            let mut use_gpu = false;

            if !config.force_cpu {
                let verdict = trainer::probe_gpu().await;
                match verdict {
                    trainer::GpuVerdict::Available { name, is_integrated, max_buffer_size, .. } => {
                        // Integrated GPU check: need buffer large enough to hold model weights.
                        // Apple Silicon (19GB+) passes easily. Intel Iris (2GB) fails.
                        // Use 4× model weight size as threshold — accounts for model + optimizer
                        // + activations in unified memory.
                        if is_integrated {
                            if max_buffer_size >= model_weight_bytes * 4 {
                                info!(
                                    "GPU '{name}' is integrated with large buffer ({}MB vs {}MB model weights) — will benchmark",
                                    max_buffer_size / (1024 * 1024), model_weight_bytes / (1024 * 1024),
                                );
                            } else {
                                info!(
                                    "GPU '{name}' is integrated with small buffer ({}MB vs {}MB model weights) — using CPU",
                                    max_buffer_size / (1024 * 1024), model_weight_bytes / (1024 * 1024),
                                );
                            }
                        }
                        if is_integrated && max_buffer_size < model_weight_bytes * 4 {
                            // skip
                        } else if max_buffer_size < model_weight_bytes {
                            info!(
                                "GPU '{name}' max buffer {}MB < model weights {}MB — using CPU",
                                max_buffer_size / (1024 * 1024), model_weight_bytes / (1024 * 1024),
                            );
                        } else {
                            use_gpu = true;
                        }
                    }
                    trainer::GpuVerdict::NoAdapter => {
                        info!("No GPU adapter — using CPU");
                    }
                }
            }

            // Calibrate batch_size + benchmark via subprocess (GPU) or memory estimate (CPU)
            if let Some(forced_bs) = config.force_batch_size {
                batch_size = forced_bs;
                grad_accum_steps = effective_batch_size / forced_bs;
                batch_calibrated = true;
                h_mini = min_h; // will be refined from first round timing
                info!("Forced batch_size={forced_bs} (skipping calibration), grad_accum={grad_accum_steps}");
            } else if use_gpu {
                info!("Calibrating GPU batch_size and timing (subprocess)...");
                let (cal_bs, cal_accum, cal_sps) = trainer::calibrate_batch_size(
                    &ckpt_path, config.seq_len, true, effective_batch_size,
                ).await;

                if cal_bs == 0 {
                    warn!("GPU batch_size calibration failed, falling back to CPU");
                    use_gpu = false;
                } else {
                    batch_size = cal_bs;
                    grad_accum_steps = cal_accum;
                    batch_calibrated = true;
                    if let Some(sps) = cal_sps {
                        gpu_secs_per_step = Some(sps);
                        h_mini = ((target_interval / sps) as u64).clamp(min_h, max_h);
                        info!("GPU calibrated: batch_size={batch_size}, grad_accum={grad_accum_steps}, {sps:.3}s/step, H_mini={h_mini}");
                    }
                }
            }

            if !use_gpu {
                config.force_cpu = true;
                if config.force_batch_size.is_none() {
                    h_mini = min_h;
                }
                info!("Using CPU, starting with H_mini = {h_mini} (will refine from first round timing)");
            }

            // CPU batch_size calibration (memory estimate)
            if config.force_cpu && !batch_calibrated {
                let (cal_bs, cal_accum, _) = trainer::calibrate_batch_size(
                    &ckpt_path, config.seq_len, false, effective_batch_size,
                ).await;
                batch_size = cal_bs.max(1);
                grad_accum_steps = cal_accum;
                batch_calibrated = true;
            }

            info!(
                "Backend decision: {} — H_mini={h_mini}, batch_size={batch_size}, grad_accum={grad_accum_steps}, effective_batch={effective_batch_size}",
                if config.force_cpu { "CPU" } else { "GPU" },
            );
            stress_tested = true;
        }

        // Compute deterministic shard assignment for this node + version
        let shard_ids = distrain_model::compute_shard_assignment(
            &node_id, version, total_shards, shards_per_node,
        );
        info!(
            "Shard assignment for v{version}: {} shards (e.g. {:?}...)",
            shard_ids.len(),
            &shard_ids[..shard_ids.len().min(5)],
        );

        // Load assigned shards (cached locally, only downloads new ones)
        let mut data_loader = data::DataLoader::from_assignment(
            &storage, &manifest, &data_cache, &shard_ids,
            config.seq_len, batch_size,
        ).await?;

        // Seek to a different position each round so re-training the same checkpoint
        // version doesn't produce identical batches.
        data_loader.seek_by_seed(seq_num);
        info!("Data loaded: {} tokens from {} assigned shards", data_loader.total_tokens(), shard_ids.len());
        resources::log_memory("after shard loading");

        // No mid-round checkpoint polling. Train the full round, push, then check
        // for new checkpoints at the top of the loop. Staleness weighting handles
        // the case where the checkpoint advanced during training.

        // Train H_mini steps
        seq_num += 1;
        let delta_path = cache_dir.join(format!("delta_{node_id}_{seq_num}.delta.zst"));

        // GPU path: use watchdog. CPU path: direct call (no driver hang risk).
        let max_mem_fraction = config.max_memory_fraction;
        let result = if !config.force_cpu {
            // GPU training with watchdog (timeout + catch_unwind)
            // force_batch_size without calibration: use generous default (no hang risk, just slow first round)
            let sps = gpu_secs_per_step.unwrap_or(if config.force_batch_size.is_some() { 300.0 } else { 10.0 });
            let mem_abort = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mem_abort_clone = mem_abort.clone();
            // Move error_buffer into watchdog, get it back after training
            let eb_in = std::mem::take(&mut error_buffer);
            let (watchdog_result, eb_out) = trainer::run_training_round_with_watchdog(
                &config,
                &ckpt_path,
                h_mini,
                &delta_path,
                &mut data_loader,
                eb_in,
                sps,
                batch_size,
                grad_accum_steps,
                move |progress| {
                    // Check memory pressure every 50 steps
                    if progress.step > 0
                        && progress.step % 50 == 0
                        && resources::check_memory_pressure(max_mem_fraction)
                    {
                        warn!(
                            "Memory pressure critical at step {}/{} — aborting round",
                            progress.step, progress.total_steps,
                        );
                        mem_abort_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                        return true;
                    }
                    false
                },
            )
            .await;
            error_buffer = eb_out;

            if mem_abort.load(std::sync::atomic::Ordering::SeqCst) {
                memory_pressure_abort = true;
            }

            match watchdog_result {
                Ok(r) => Ok(r),
                Err(trainer::TrainingFailure::GpuHung { timeout_secs }) => {
                    if config.force_batch_size.is_some() {
                        // force_batch_size means user trusts GPU; retry instead of falling back
                        warn!(
                            "GPU hung during training (timeout {timeout_secs:.0}s) — retrying on GPU (force_batch_size set)"
                        );
                        continue;
                    }
                    warn!(
                        "GPU hung during training (timeout {timeout_secs:.0}s) — falling back to CPU permanently"
                    );
                    config.force_cpu = true;
                    h_mini = config.min_inner_steps;
                    // Re-calibrate batch_size for CPU
                    let (cal_bs, cal_accum, _) = trainer::calibrate_batch_size(
                        &ckpt_path, config.seq_len, false, effective_batch_size,
                    ).await;
                    batch_size = cal_bs.max(1);
                    grad_accum_steps = cal_accum;
                    info!("Falling back to CPU — H_mini={h_mini}, batch_size={batch_size}, grad_accum={grad_accum_steps}");
                    continue; // Retry same checkpoint on CPU
                }
                Err(trainer::TrainingFailure::GpuPanic { message }) => {
                    // OOM recovery: halve batch_size, double grad_accum
                    if batch_size > 1 {
                        let old_bs = batch_size;
                        batch_size /= 2;
                        grad_accum_steps = effective_batch_size / batch_size;
                        warn!(
                            "GPU panic at batch_size={old_bs}: {message} — recovering to batch_size={batch_size}, grad_accum={grad_accum_steps}"
                        );
                        continue; // Retry round with smaller batch
                    }
                    // batch_size=1 and still failing — fall back to CPU
                    warn!(
                        "GPU panicked at batch_size=1: {message} — falling back to CPU permanently"
                    );
                    config.force_cpu = true;
                    h_mini = config.min_inner_steps;
                    let (cal_bs, cal_accum, _) = trainer::calibrate_batch_size(
                        &ckpt_path, config.seq_len, false, effective_batch_size,
                    ).await;
                    batch_size = cal_bs.max(1);
                    grad_accum_steps = cal_accum;
                    info!("Falling back to CPU — H_mini={h_mini}, batch_size={batch_size}, grad_accum={grad_accum_steps}");
                    continue; // Retry same checkpoint on CPU
                }
                Err(trainer::TrainingFailure::Other(e)) => Err(e),
            }
        } else {
            // CPU training — no watchdog needed
            let mem_abort = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let mem_abort_clone = mem_abort.clone();
            let r = trainer::run_training_with_progress(
                &config,
                &ckpt_path,
                h_mini,
                &delta_path,
                &mut data_loader,
                &mut error_buffer,
                batch_size,
                grad_accum_steps,
                move |progress| {
                    if progress.step > 0
                        && progress.step % 50 == 0
                        && resources::check_memory_pressure(max_mem_fraction)
                    {
                        warn!(
                            "Memory pressure critical at step {}/{} — aborting round",
                            progress.step, progress.total_steps,
                        );
                        mem_abort_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                        return true;
                    }
                    false
                },
            )
            .await;

            if mem_abort.load(std::sync::atomic::Ordering::SeqCst) {
                memory_pressure_abort = true;
            }
            r
        }?;

        last_trained_version = Some(version);
        result_elapsed = result.elapsed_secs;
        info!(
            "Training done: {}/{} steps, loss={:.4}, tokens={}, time={:.1}s, batch_size={}, grad_accum={}, effective_batch={}",
            result.steps_completed, h_mini, result.final_loss, result.tokens_processed, result.elapsed_secs,
            result.batch_size, grad_accum_steps, effective_batch_size,
        );

        // Refine H_mini from actual training speed (first round or after fallback)
        if result.steps_completed > 0 && result.elapsed_secs > 0.0 {
            let actual_sps = result.elapsed_secs / result.steps_completed as f64;
            let refined = ((config.target_push_interval_secs / actual_sps) as u64)
                .clamp(config.min_inner_steps, config.max_inner_steps);
            if refined != h_mini {
                info!("Refining H_mini: {h_mini} → {refined} (measured {actual_sps:.1}s/step)");
                h_mini = refined;
            }
        }

        // Don't push if only warmup steps completed (lr=0, delta is noise)
        let warmup_steps = (h_mini as f64 * 0.2).max(2.0) as u64;
        if result.steps_completed <= warmup_steps {
            warn!(
                "Only {}/{} steps completed (warmup={}) — skipping push (delta would be noise)",
                result.steps_completed, h_mini, warmup_steps,
            );
            let _ = tokio::fs::remove_file(&delta_path).await;
            continue;
        }

        // Upload delta to R2 (retry with exponential backoff)
        let delta_key =
            distrain_shared::paths::delta_path(version, &node_id, seq_num);
        let mut upload_ok = false;
        for attempt in 1..=5 {
            match storage.upload_from_file(&delta_key, &delta_path).await {
                Ok(()) => {
                    upload_ok = true;
                    break;
                }
                Err(e) => {
                    warn!("Delta upload failed (attempt {attempt}/5): {e:#}");
                    if attempt < 5 {
                        let backoff = std::time::Duration::from_secs(2u64.pow(attempt));
                        info!("Retrying upload in {}s...", backoff.as_secs());
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
        if !upload_ok {
            warn!("Failed to upload delta after 5 attempts, skipping push");
            let _ = tokio::fs::remove_file(&delta_path).await;
            continue;
        }

        // Push metadata to coordinator (retry with exponential backoff)
        let push = distrain_shared::types::DeltaPush {
            node_id: distrain_shared::types::NodeId(node_id.clone()),
            seq_num,
            checkpoint_version: version,
            inner_steps: result.steps_completed,
            delta_key: delta_key.clone(),
            training_loss: result.final_loss,
            tokens_processed: result.tokens_processed,
            training_time_secs: result.elapsed_secs,
            compressed_bytes: result.compression_stats.as_ref().map(|s| s.compressed_bytes),
            dense_norm: result.compression_stats.as_ref().map(|s| s.dense_norm),
            sparse_norm: result.compression_stats.as_ref().map(|s| s.sparse_norm),
        };

        let mut push_ok = false;
        for attempt in 1..=5 {
            match coordinator.push_delta(&push).await {
                Ok(resp) => {
                    if resp.accepted {
                        info!("Push accepted (ckpt v{})", resp.checkpoint_version);
                    } else {
                        info!(
                            "Push rejected: {}",
                            resp.reason.unwrap_or_else(|| "unknown".to_string())
                        );
                    }
                    push_ok = true;
                    break;
                }
                Err(e) => {
                    warn!("Push to coordinator failed (attempt {attempt}/5): {e:#}");
                    if attempt < 5 {
                        let backoff = std::time::Duration::from_secs(2u64.pow(attempt));
                        info!("Retrying push in {}s...", backoff.as_secs());
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
        if !push_ok {
            warn!("Failed to push delta after 5 attempts, continuing to next round");
        }

        // Append per-round metrics to local JSONL (best-effort, for paper analysis)
        {
            use tokio::io::AsyncWriteExt;
            let metrics_path = cache_dir.join("node_metrics.jsonl");
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let entry = serde_json::json!({
                "timestamp": timestamp,
                "node_id": &node_id,
                "seq": seq_num,
                "version": version,
                "steps": result.steps_completed,
                "loss": result.final_loss,
                "tokens": result.tokens_processed,
                "secs": result.elapsed_secs,
                "h_mini": h_mini,
                "batch_size": result.batch_size,
                "compressed_bytes": result.compression_stats.as_ref().map(|s| s.compressed_bytes),
                "raw_bytes": result.compression_stats.as_ref().map(|s| s.raw_param_bytes),
                "retention": result.compression_stats.as_ref().map(|s| s.retention_ratio),
                "top_k": result.compression_stats.as_ref().map(|s| s.top_k_fraction),
                "dense_norm": result.compression_stats.as_ref().map(|s| s.dense_norm),
                "sparse_norm": result.compression_stats.as_ref().map(|s| s.sparse_norm),
            });
            if let Ok(line) = serde_json::to_string(&entry) {
                if let Ok(mut f) = tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&metrics_path)
                    .await
                {
                    let _ = f.write_all(format!("{line}\n").as_bytes()).await;
                }
            }
        }

        // Cleanup delta file
        let _ = tokio::fs::remove_file(&delta_path).await;

        // Housekeeping: keep only the 3 most recent checkpoints in cache
        cleanup_old_checkpoints(&cache_dir, 3).await;
    }
}

/// Delete old cached checkpoints, keeping only the N most recent.
async fn cleanup_old_checkpoints(cache_dir: &std::path::Path, keep: usize) {
    let mut ckpts: Vec<(u64, std::path::PathBuf)> = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(cache_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('v') && name.ends_with("_model.safetensors") {
                if let Some(ver_str) = name.strip_prefix('v').and_then(|s| s.strip_suffix("_model.safetensors")) {
                    if let Ok(ver) = ver_str.parse::<u64>() {
                        ckpts.push((ver, entry.path()));
                    }
                }
            }
        }
    }

    if ckpts.len() <= keep {
        return;
    }

    ckpts.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
    for (ver, path) in &ckpts[keep..] {
        if tokio::fs::remove_file(path).await.is_ok() {
            info!("Housekeeping: deleted cached checkpoint v{ver}");
        }
    }
}

async fn run_bootstrap(config: &NodeConfig, preset_name: &str) -> Result<()> {
    use distrain_model::checkpoint::save_state_dict_safetensors_bytes;
    use distrain_model::config::ModelPreset;
    use distrain_model::model::DistrainTransformerModule;
    use distrain_model::{CpuBackend, CpuDevice};

    let preset = ModelPreset::from_name(preset_name)
        .with_context(|| format!("Unknown preset: {preset_name}. Use: micro-test, tiny, small, medium, large"))?;

    let model_config = preset.config();
    let param_count = model_config.param_count();
    info!(
        "Bootstrapping v0 checkpoint: {preset_name} ({param_count} params, {}h/{}L/{}v)",
        model_config.hidden_dim, model_config.num_layers, model_config.vocab_size,
    );

    let device: CpuDevice = Default::default();
    let module = DistrainTransformerModule::<CpuBackend>::new(&model_config, &device);
    let state_dict = module.extract_state_dict();
    let shapes = module.extract_shapes();

    info!("Serializing checkpoint to safetensors...");
    let bytes = save_state_dict_safetensors_bytes(&state_dict, &shapes)
        .context("Failed to serialize checkpoint")?;
    info!("Checkpoint size: {} bytes ({:.1} MB)", bytes.len(), bytes.len() as f64 / 1e6);

    // Upload to R2
    let storage = Storage::new(&config.storage).await?;
    storage.ensure_bucket().await?;

    let key = distrain_shared::paths::checkpoint_path(0);
    info!("Uploading to R2: {key}");
    storage.put(&key, bytes).await?;

    // Upload metadata
    let metadata = serde_json::json!({
        "version": 0,
        "preset": preset_name,
        "param_count": param_count,
        "hidden_dim": model_config.hidden_dim,
        "num_layers": model_config.num_layers,
        "vocab_size": model_config.vocab_size,
    });
    let meta_key = distrain_shared::paths::checkpoint_metadata_path(0);
    storage.put_json(&meta_key, &metadata).await?;

    // Initialize accumulator state
    let acc = serde_json::json!({
        "checkpoint_version": 0,
        "contributions": [],
        "version": 0,
    });
    let acc_key = distrain_shared::paths::accumulator_path();
    storage.put_json(&acc_key, &acc).await?;

    info!("Bootstrap complete. v0 checkpoint uploaded to R2.");
    Ok(())
}

async fn run_benchmark(config: &NodeConfig) -> Result<()> {
    // Show GPU probe result
    let verdict = trainer::probe_gpu().await;
    info!("GPU verdict: {verdict:?}");

    let (h_mini, use_cpu) = trainer::calibrate(config).await?;
    info!("Benchmark complete: H_mini = {h_mini} (backend: {})", if use_cpu { "CPU" } else { "GPU" });
    Ok(())
}

async fn show_config(config: &NodeConfig) -> Result<()> {
    let local = config.training_params.clone().unwrap_or_default();
    println!("Local training params (from config file):");
    print_params(&local);

    // Try to fetch from coordinator
    let coordinator = client::CoordinatorClient::new(&config.coordinator_url);
    match coordinator.register(config, None).await {
        Ok(reg) => {
            if let Some(remote) = reg.training_params {
                println!("\nCoordinator training params:");
                print_params(&remote);
            } else {
                println!("\nCoordinator did not send training params (using local defaults).");
            }
        }
        Err(e) => {
            println!("\nCould not reach coordinator: {e:#}");
            println!("Using local defaults.");
        }
    }
    Ok(())
}

fn print_params(p: &distrain_shared::types::TrainingParams) {
    println!("  batch_size:       {}", p.batch_size);
    println!("  seq_len:          {}", p.seq_len);
    println!("  lr_max:           {:.2e}", p.lr_max);
    println!("  lr_min:           {:.2e}", p.lr_min);
    println!("  weight_decay:     {}", p.weight_decay);
    println!("  grad_clip_norm:   {}", p.grad_clip_norm);
    println!("  warmup_fraction:  {:.0}%", p.warmup_fraction * 100.0);
    println!("  shards_fraction:  {:.0}%", p.shards_fraction * 100.0);
}

/// Child process: run GPU stress test and print secs_per_step to stdout.
/// Exit 0 = success, non-zero = failure. If the GPU crashes the OS, parent survives.
async fn run_gpu_stress_test_child(checkpoint: &str, batch_size: usize, seq_len: usize) -> Result<()> {
    use distrain_model::checkpoint::load_safetensors_map;
    use distrain_model::model::{compute_lm_loss, precompute_rope_tables, DistrainTransformerModule};
    use distrain_model::{GpuBackend, GpuDevice};
    use burn::grad_clipping::GradientClippingConfig;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
    use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
    use std::time::Instant;

    let ckpt_path = std::path::PathBuf::from(checkpoint);
    let device: GpuDevice = Default::default();

    let start_params = load_safetensors_map(&ckpt_path)?;
    let model_config = trainer::infer_model_config(&ckpt_path)?;
    let (rope_cos, rope_sin) = precompute_rope_tables::<GpuBackend>(
        model_config.head_dim(), model_config.max_seq_len, model_config.rope_theta, &device,
    );
    let module = DistrainTransformerModule::<GpuBackend>::new(&model_config, &device);
    let mut module = module.load_state_dict(&start_params, &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<GpuBackend, DistrainTransformerModule<GpuBackend>>();

    let start = Instant::now();
    let tokens: Vec<i64> = (0..(batch_size * seq_len))
        .map(|i| (trainer::splitmix64_pub(i as u64) % model_config.vocab_size as u64) as i64)
        .collect();
    let batch = Tensor::<GpuBackend, 2, Int>::from_data(
        TensorData::new(tokens, [batch_size, seq_len]), &device,
    );
    let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
    let loss_val: f64 = loss.clone().into_scalar().elem();

    if loss_val == 0.0 || loss_val.is_nan() || loss_val.is_infinite() || loss_val < 0.0 {
        anyhow::bail!("GPU produced invalid loss={loss_val}");
    }

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &module);
    let _module = optim.step(3e-4, module, grads_params);

    let secs_per_step = start.elapsed().as_secs_f64();
    // Print result for parent to parse
    println!("GPU_STRESS_OK {secs_per_step:.6}");
    Ok(())
}

async fn run_generate(
    checkpoint_path: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
) -> Result<()> {
    use distrain_model::checkpoint::load_safetensors_map;
    use distrain_model::model::{precompute_rope_tables, DistrainTransformerModule};
    use distrain_model::{GpuBackend, GpuDevice};
    use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
    use tokenizers::Tokenizer;

    let ckpt_path = std::path::PathBuf::from(checkpoint_path);
    let model_config = trainer::infer_model_config(&ckpt_path)?;
    let vocab_size = model_config.vocab_size;

    // Load Mistral v0.3 tokenizer (32768 vocab, IDs natively in [0, 32768))
    let tokenizer_bytes = include_bytes!("../../../tokenizers/mistral-v0.3.json");
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Encode prompt — no modulo mapping needed
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {e}"))?;
    let mut tokens: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

    if tokens.is_empty() {
        tokens.push(0); // BOS-like fallback
    }

    // Load model on GPU
    let device: GpuDevice = Default::default();
    let start_params = load_safetensors_map(&ckpt_path)?;
    let (rope_cos, rope_sin) = precompute_rope_tables::<GpuBackend>(
        model_config.head_dim(),
        model_config.max_seq_len,
        model_config.rope_theta,
        &device,
    );
    let module = DistrainTransformerModule::<GpuBackend>::new(&model_config, &device);
    let module = module.load_state_dict(&start_params, &device);

    // Track all token IDs for full-sequence decode (preserves spaces)
    let all_token_ids: &mut Vec<u32> = &mut tokens.iter().map(|&t| t as u32).collect();
    let prompt_decoded = tokenizer.decode(all_token_ids.as_slice(), true).unwrap_or_default();
    print!("{prompt_decoded}");
    let mut printed_len = prompt_decoded.len();

    // Autoregressive generation
    for _ in 0..max_tokens {
        let seq_len = tokens.len().min(model_config.max_seq_len);
        let input_tokens = &tokens[tokens.len() - seq_len..];

        let batch = Tensor::<GpuBackend, 2, Int>::from_data(
            TensorData::new(input_tokens.to_vec(), [1, seq_len]),
            &device,
        );

        // Forward pass through model components
        let mut h = module.embedding.encode(batch);
        for layer in &module.layers {
            h = layer.forward(h, &rope_cos, &rope_sin, 0);
        }
        h = module.final_norm.forward(h);
        let logits = module.embedding.decode(h); // [1, seq, vocab]

        // Get logits for last position
        let [_b, s, _v] = logits.dims();
        let last_logits = logits.slice([0..1, (s - 1)..s, 0..vocab_size]);
        let last_logits = last_logits.reshape([vocab_size]); // [vocab]

        // Sample next token
        let next_token = if temperature <= 0.0 {
            // Greedy
            let data = last_logits.into_data();
            let values: Vec<f32> = data.to_vec().unwrap();
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as i64
        } else {
            // Temperature sampling
            let scaled = last_logits.div_scalar(temperature as f32);
            // Softmax
            let max_val: f32 = scaled.clone().max().into_scalar().elem();
            let shifted = scaled.sub_scalar(max_val);
            let exp = shifted.exp();
            let sum: f32 = exp.clone().sum().into_scalar().elem();
            let probs = exp.div_scalar(sum);

            let data = probs.into_data();
            let prob_vec: Vec<f32> = data.to_vec().unwrap();

            // Weighted random sample
            let r: f32 = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as f32)
                / 4_294_967_296.0;

            let mut cumulative = 0.0f32;
            let mut chosen = 0usize;
            for (i, &p) in prob_vec.iter().enumerate() {
                cumulative += p;
                if cumulative > r {
                    chosen = i;
                    break;
                }
            }
            chosen as i64
        };

        tokens.push(next_token);
        all_token_ids.push(next_token as u32);

        // EOS stop (Mistral v0.3 EOS = 2)
        if next_token == 2 {
            break;
        }

        // Full-sequence decode to preserve spaces between tokens
        let full = tokenizer.decode(all_token_ids.as_slice(), true).unwrap_or_default();
        print!("{}", &full[printed_len..]);
        printed_len = full.len();
        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    println!();
    Ok(())
}

async fn run_eval(config: &NodeConfig, checkpoints: &[String], num_batches: u64) -> Result<()> {
    use distrain_model::checkpoint::load_safetensors_map;
    use distrain_model::model::{compute_lm_loss, precompute_rope_tables, DistrainTransformerModule};
    use distrain_model::{GpuBackend, GpuDevice};
    use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

    let storage = Storage::new(&config.storage).await?;
    let cache_dir = shellexpand::tilde(&config.cache_dir).to_string();
    let cache_dir = std::path::PathBuf::from(&cache_dir);
    tokio::fs::create_dir_all(&cache_dir).await?;

    // Load one shard of real data for evaluation
    let (manifest, data_cache) = data::DataLoader::load_manifest(&storage, &cache_dir).await?;
    let mut data_loader = data::DataLoader::from_assignment(
        &storage, &manifest, &data_cache, &[0],
        config.seq_len, config.batch_size,
    ).await?;

    let device: GpuDevice = Default::default();

    println!("Evaluating {} checkpoint(s) on {} batches of real data (GPU)...\n", checkpoints.len(), num_batches);

    for ckpt_path_str in checkpoints {
        let ckpt_path = std::path::PathBuf::from(ckpt_path_str);
        if !ckpt_path.exists() {
            println!("{ckpt_path_str}: FILE NOT FOUND");
            continue;
        }

        let model_config = trainer::infer_model_config(&ckpt_path)?;
        let start_params = load_safetensors_map(&ckpt_path)?;
        let (rope_cos, rope_sin) = precompute_rope_tables::<GpuBackend>(
            model_config.head_dim(),
            model_config.max_seq_len,
            model_config.rope_theta,
            &device,
        );
        let module = DistrainTransformerModule::<GpuBackend>::new(&model_config, &device);
        let module = module.load_state_dict(&start_params, &device);

        // Reset data loader to same position for fair comparison
        data_loader.reset();

        let mut total_loss = 0.0;
        let mut count = 0u64;
        for _ in 0..num_batches {
            let tokens = data_loader.next_batch();
            let batch = Tensor::<GpuBackend, 2, Int>::from_data(
                TensorData::new(tokens, [config.batch_size, config.seq_len]),
                &device,
            );
            let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
            let loss_val: f64 = loss.into_scalar().elem();
            if !loss_val.is_nan() {
                total_loss += loss_val;
                count += 1;
            }
        }

        let avg_loss = if count > 0 { total_loss / count as f64 } else { f64::NAN };
        println!("{ckpt_path_str}: avg_loss = {avg_loss:.4} ({count}/{num_batches} valid batches)");
    }

    Ok(())
}

async fn show_status(config: &NodeConfig) -> Result<()> {
    let coordinator = client::CoordinatorClient::new(&config.coordinator_url);
    let status = coordinator.get_status().await?;
    println!("Checkpoint version: {}", status.checkpoint_version);
    println!("Active nodes: {}", status.active_nodes);
    println!("Accumulator contributions: {}", status.accumulator_contributions);
    if let Some(loss) = status.latest_val_loss {
        println!("Latest val loss: {loss:.4}");
    }
    Ok(())
}
