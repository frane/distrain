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
use distrain_shared::config::{NodeConfig, StorageConfig};
use distrain_shared::storage::Storage;
use tracing::{error, info, warn};

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
        /// Auto-discover config from coordinator URL (no node.toml needed)
        #[arg(long)]
        coordinator_url: Option<String>,
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
    /// Single-GPU baseline training (no coordinator, no compression, no async merge)
    /// For paper comparison: same model, same data, same hyperparameters.
    Baseline {
        /// Path to checkpoint file to start from
        checkpoint: String,
        /// Path to data directory with shard files
        #[arg(long)]
        data_dir: String,
        /// Number of training steps
        #[arg(long, default_value = "1000")]
        steps: u64,
        /// Output JSONL file for per-step metrics
        #[arg(long, default_value = "baseline.jsonl")]
        output: String,
        /// Force CPU backend
        #[arg(long)]
        cpu: bool,
        /// Batch size
        #[arg(long, default_value = "4")]
        batch_size: usize,
        /// Sequence length
        #[arg(long, default_value = "512")]
        seq_len: usize,
        /// Learning rate max
        #[arg(long, default_value = "0.0003")]
        lr_max: f64,
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
        Commands::Start { config, cpu, coordinator_url } => {
            let mut config = if let Some(ref url) = coordinator_url {
                build_config_from_coordinator(url).await?
            } else {
                load_config(&config)?
            };
            config.force_cpu = cpu || config.gpu_device < 0;
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
        Commands::Baseline {
            checkpoint,
            data_dir,
            steps,
            output,
            cpu,
            batch_size,
            seq_len,
            lr_max,
        } => {
            run_baseline(&checkpoint, &data_dir, steps, &output, cpu, batch_size, seq_len, lr_max).await?;
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

async fn build_config_from_coordinator(url: &str) -> Result<NodeConfig> {
    info!("Auto-discovering config from coordinator: {url}");
    let client = client::CoordinatorClient::new(url);
    let auto_config = client.get_config().await
        .context("Failed to fetch auto-config from coordinator")?;

    info!(
        "Got config: coordinator v{}, bucket={}, training lr={:.2e}",
        auto_config.coordinator_version,
        auto_config.storage.bucket,
        auto_config.training_params.lr_max,
    );

    let storage = StorageConfig {
        endpoint: auto_config.storage.endpoint,
        bucket: auto_config.storage.bucket,
        access_key_id: auto_config.storage.access_key_id,
        secret_access_key: auto_config.storage.secret_access_key,
        region: auto_config.storage.region,
    };

    Ok(NodeConfig {
        coordinator_url: url.to_string(),
        api_key: String::new(),
        storage,
        gpu_device: 0,
        target_push_interval_secs: auto_config.training_params.target_push_interval_secs,
        min_inner_steps: auto_config.training_params.min_inner_steps,
        max_inner_steps: auto_config.training_params.max_inner_steps,
        cache_dir: "~/.distrain/cache".to_string(),
        max_cache_gb: 100,
        batch_size: Some(auto_config.training_params.batch_size),
        seq_len: auto_config.training_params.seq_len,
        training_params: Some(auto_config.training_params),
        max_memory_fraction: 0.80,
        force_batch_size: None,
        ..Default::default()
    })
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

    // Phase 1: Quick GPU probe — used for hardware profile + early adapter check.
    // No model loading, no calibration. Actual backend + H_mini computed in Phase 2.
    let gpu_verdict = if !config.force_cpu {
        let verdict = trainer::probe_gpu().await;
        match &verdict {
            trainer::GpuVerdict::Available { name, .. } => {
                info!("GPU adapter found: {name} — will benchmark in Phase 2");
            }
            trainer::GpuVerdict::NoAdapter => {
                info!("No GPU adapter — will use CPU");
                config.force_cpu = true;
            }
        }
        Some(verdict)
    } else {
        None
    };

    // Build hardware profile from GPU probe + system info
    let hardware_profile = {
        use distrain_shared::types::{DeviceType, HardwareProfile};
        let sys = sysinfo::System::new_all();
        let cpu_cores = sys.cpus().len() as u32;
        let total_mem = sys.total_memory();
        let ram_mb = total_mem / (1024 * 1024);

        match &gpu_verdict {
            Some(trainer::GpuVerdict::Available { name, is_integrated, vram_mb, .. }) => {
                let device_type = if *is_integrated { DeviceType::IntegratedGpu } else { DeviceType::DiscreteGpu };
                HardwareProfile {
                    gpu_model: name.clone(),
                    vram_mb: vram_mb.unwrap_or(0),
                    device_type,
                    cpu_cores,
                    ram_mb,
                    ..Default::default()
                }
            }
            _ => {
                HardwareProfile {
                    gpu_model: "none".to_string(),
                    vram_mb: 0,
                    device_type: DeviceType::Cpu,
                    cpu_cores,
                    ram_mb,
                    ..Default::default()
                }
            }
        }
    };
    info!(
        "Hardware profile: {} ({:?}, {} MiB VRAM, {} cores, {} MiB RAM)",
        hardware_profile.gpu_model, hardware_profile.device_type,
        hardware_profile.vram_mb, hardware_profile.cpu_cores, hardware_profile.ram_mb,
    );

    // Don't register yet. Node registers AFTER calibration with full hardware profile.
    // Training params come from /config endpoint (auto-discovery) or node.toml.
    // Node ID comes from persistent file or will be assigned at registration.
    let mut node_id = persistent_id.unwrap_or_default();
    let mut seq_num: u64 = 0;

    // H_mini temporary value — overridden by coordinator params below
    let mut h_mini: u64 = config.min_inner_steps;

    // Cache dir
    let cache_dir = shellexpand::tilde(&config.cache_dir).to_string();
    let cache_dir = std::path::PathBuf::from(cache_dir);
    tokio::fs::create_dir_all(&cache_dir).await?;

    // Load data manifest from R2 (shard list — actual data loaded per round)
    // Retry: network may not be ready immediately on container start.
    info!("Loading data manifest from storage...");
    let (manifest, data_cache) = {
        let mut result = None;
        for attempt in 1..=10u32 {
            match data::DataLoader::load_manifest(&storage, &cache_dir).await {
                Ok(r) => { result = Some(r); break; }
                Err(e) => {
                    if attempt < 10 {
                        warn!("Manifest download failed (attempt {attempt}/10): {e:#}. Retrying in {attempt}s...");
                        tokio::time::sleep(std::time::Duration::from_secs(attempt as u64)).await;
                    } else {
                        anyhow::bail!("Failed to load data manifest after 10 attempts: {e:#}");
                    }
                }
            }
        }
        result.unwrap()
    };

    let total_shards = manifest.shards.len();
    let params = config
        .training_params
        .as_ref()
        .cloned()
        .unwrap_or_default();
    let requested_shards = params.shards_per_node(total_shards);
    // Use centralized training params, but allow local node.toml to cap H_mini
    // (e.g., thermally limited devices like Intel laptops)
    let min_h = if config.max_inner_steps < params.min_inner_steps {
        config.max_inner_steps  // local cap takes precedence
    } else {
        params.min_inner_steps
    };
    let max_h = config.max_inner_steps.min(params.max_inner_steps);
    let target_interval = params.target_push_interval_secs;
    info!("Data manifest: {total_shards} shards, requested {requested_shards} per round");
    h_mini = min_h; // use coordinator-provided value
    info!("Training params: H_mini=[{min_h}, {max_h}], push_interval={target_interval}s");

    // Will be clamped by memory budget once we know the model size (first checkpoint)
    let mut shards_per_node = requested_shards;
    let mut budget_computed = false;
    // Measured secs/step for the real model on GPU — set by stress test, used by watchdog.
    let gpu_secs_per_step: Option<f64> = None;
    // Whether the GPU stress test has been run (only once, on first iteration).
    let mut stress_tested = false;
    // Whether the last round was aborted due to memory pressure.
    let mut memory_pressure_abort = false;
    // Measured upload bandwidth (bytes/sec), used for bandwidth-aware top-k adaptation.
    // Updated from shared Arc after awaiting background upload.
    #[allow(unused_variables, unused_assignments)]
    let mut measured_upload_bps: Option<f64> = None;
    // Persistent error buffer for compression error feedback across rounds.
    let mut error_buffer = distrain_model::compression::ErrorBuffer::new();

    // Try to resume from saved state
    if let Ok(Some(saved_state)) = distrain_node::resume::NodeState::load(&cache_dir) {
        info!(
            "Resuming from saved state: v{}, seq={}, node={}",
            saved_state.last_checkpoint_version, saved_state.seq_num, saved_state.node_id,
        );
        seq_num = saved_state.seq_num;
        if node_id.is_empty() {
            node_id = saved_state.node_id;
        }
    }
    // Try to load saved error buffer
    match distrain_node::resume::load_error_buffer(&cache_dir) {
        Ok(Some(eb)) => {
            info!("Resumed error buffer ({} tensors)", eb.buffer.len());
            error_buffer = eb;
        }
        Ok(None) => {}
        Err(e) => warn!("Failed to load error buffer (starting fresh): {e}"),
    }
    // Last round's elapsed time (for poll delay estimation).
    let mut _result_elapsed: f64 = 0.0;
    // Auto-calibrated batch size and gradient accumulation steps.
    // If user specified batch_size (via toml or force_batch_size), use it.
    // Otherwise auto-detect from VRAM during calibration.
    let user_batch_size = config.batch_size.or(config.force_batch_size);
    let mut effective_batch_size = user_batch_size.unwrap_or(4); // provisional, updated by auto-detect
    let mut batch_size = effective_batch_size;
    let mut grad_accum_steps: usize = 1;
    let mut batch_calibrated = false;
    // Adaptive inner LR: node observes own loss stability and adjusts
    let base_lr_max = params.lr_max;
    let mut effective_lr_max = base_lr_max;
    // Track checkpoint version to clear error buffer on version change
    let mut last_trained_version: Option<u64> = None;

    // GPU pipeline: overlap upload with next training round
    let mut pending_upload: Option<tokio::task::JoinHandle<Result<()>>> = None;
    // Shared measured upload bandwidth — written by background upload task, read by main loop
    let measured_upload_bps_shared = std::sync::Arc::new(std::sync::Mutex::new(None::<f64>));

    // Graceful shutdown: SIGINT/SIGTERM saves state before exit
    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let shutdown_flag = shutdown.clone();
        let cache_dir_sig = cache_dir.clone();
        tokio::spawn(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to register SIGTERM handler");
            #[cfg(unix)]
            tokio::select! {
                _ = ctrl_c => info!("Received SIGINT — saving state and shutting down"),
                _ = sigterm.recv() => info!("Received SIGTERM — saving state and shutting down"),
            }
            #[cfg(not(unix))]
            {
                let _ = ctrl_c.await;
                info!("Received SIGINT — saving state and shutting down");
            }
            // Mark shutdown — the main loop checks this between rounds
            shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);
            // Give the main loop a moment to save state, then force exit
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            warn!("Shutdown timeout — forcing exit");
            // Save a minimal state.toml as a safety net
            let state = distrain_node::resume::NodeState {
                last_checkpoint_version: 0,
                seq_num: 0,
                shard_index: 0,
                shard_offset: 0,
                node_id: String::new(),
                saved_at: chrono::Utc::now().to_rfc3339(),
            };
            let _ = state.save(&cache_dir_sig);
            std::process::exit(0);
        });
    }

    loop {
        // Await previous background upload before starting new round
        if let Some(handle) = pending_upload.take() {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!("Previous delta upload failed: {e}"),
                Err(e) => warn!("Previous upload task panicked: {e}"),
            }
            // Sync measured bandwidth from background task
            measured_upload_bps = *measured_upload_bps_shared.lock().unwrap();
        }
        // Check for graceful shutdown between rounds
        if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
            info!("Shutdown requested — saving final state");
            let node_state = distrain_node::resume::NodeState {
                last_checkpoint_version: last_trained_version.unwrap_or(0),
                seq_num,
                shard_index: 0,
                shard_offset: 0,
                node_id: node_id.clone(),
                saved_at: chrono::Utc::now().to_rfc3339(),
            };
            if let Err(e) = node_state.save(&cache_dir) {
                warn!("Failed to save state on shutdown: {e}");
            }
            if let Err(e) = distrain_node::resume::save_error_buffer(&error_buffer, &cache_dir) {
                warn!("Failed to save error buffer on shutdown: {e}");
            }
            info!("State saved. Exiting.");
            return Ok(());
        }

        // Dynamic shard reduction: if previous round hit memory pressure, halve shards
        if memory_pressure_abort {
            let prev = shards_per_node;
            shards_per_node = (shards_per_node / 2).max(1);
            warn!(
                "Memory pressure detected last round — reducing shards from {prev} to {shards_per_node}"
            );
            memory_pressure_abort = false;
        }

        // Heartbeat: tell coordinator we're alive (only after registration)
        if !node_id.is_empty() {
            match coordinator.heartbeat(&node_id, None, None, None, None).await {
                Ok(resp) => info!("Heartbeat OK ({} active nodes)", resp.active_nodes),
                Err(e) => warn!("Heartbeat failed (non-fatal): {e}"),
            }
        }

        // Get latest checkpoint version
        let ckpt_info = coordinator.get_latest_checkpoint().await?;
        let version = ckpt_info.version;
        info!("Latest checkpoint: v{version}");

        // Clear error buffer when checkpoint version changes — residuals are relative
        // to old weights and no longer valid for the new checkpoint.
        if let Some(prev_v) = last_trained_version {
            if version != prev_v {
                // Keep error buffer on checkpoint change (100% retention).
                // The accumulated error is valid gradient signal from prior
                // training rounds — decaying it loses information permanently.
                info!("Checkpoint advanced v{prev_v} → v{version} — keeping error buffer (100% retention)");
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
            let model_config = trainer::infer_model_config(&ckpt_path)?;
            // Model weights in BF16 (2 bytes/param) — this is what the GPU actually loads
            let model_weight_bytes = model_config.param_count() as u64 * 2;

            let mut use_gpu = false;
            let mut gpu_vram_mb: Option<u64> = None;

            if !config.force_cpu {
                let verdict = trainer::probe_gpu().await;
                match verdict {
                    trainer::GpuVerdict::Available { name, is_integrated, max_buffer_size, vram_mb, .. } => {
                        // Use VRAM if known, else estimate from max_buffer_size (integrated GPUs)
                        gpu_vram_mb = vram_mb.or_else(|| {
                            // max_buffer_size is in bytes, convert to MiB
                            // For integrated GPUs, buffer size ≈ usable GPU memory
                            Some(max_buffer_size / (1024 * 1024))
                        });

                        // Integrated GPU with tiny buffer: skip (e.g., Intel Iris 2GB)
                        // Discrete GPUs and large integrated GPUs: always try, let probe decide.
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
            if let Some(forced_bs) = user_batch_size {
                batch_size = forced_bs;
                effective_batch_size = forced_bs;
                grad_accum_steps = 1;
                batch_calibrated = true;
                h_mini = min_h; // will be refined from first round timing
                info!("Forced batch_size={forced_bs} (skipping calibration), grad_accum={grad_accum_steps}");
            } else if use_gpu {
                // Batch size determined during real training startup in continuous.rs:
                // after model+optimizer load, query actual free VRAM, compute batch from that.
                // Just use VRAM estimate as initial value — continuous training will adjust.
                batch_size = if let Some(vram) = gpu_vram_mb {
                    trainer::estimate_batch_size_from_model(vram, &model_config, config.seq_len)
                } else {
                    4
                };
                effective_batch_size = batch_size;
                grad_accum_steps = 1;
                batch_calibrated = true;
                h_mini = min_h;
                info!("Initial batch_size={batch_size} from VRAM estimate. Continuous training will measure and adjust.");
            }

            if !use_gpu {
                config.force_cpu = true;
                if user_batch_size.is_none() {
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

            // Register with coordinator NOW — after autotune + calibration.
            // This is the only registration. No early registration means no ghost nodes.
            let sps = gpu_secs_per_step.unwrap_or(1.0); // estimate if timing probe was skipped
            let overhead_secs = 15.0;
            let expected_round = h_mini as f64 * sps + overhead_secs;
            let mut calibrated_profile = hardware_profile.clone();
            calibrated_profile.step_time_secs = gpu_secs_per_step;
            calibrated_profile.h_mini = Some(h_mini);
            calibrated_profile.batch_size = Some(batch_size);
            calibrated_profile.expected_round_time_secs = Some(expected_round);
            let persistent = if node_id.is_empty() { None } else { Some(node_id.clone()) };
            let reg = coordinator.register(&config, persistent, Some(calibrated_profile)).await?;
            node_id = reg.node_id.0.clone();
            info!(
                "Registered as {node_id}: step_time={sps:.2}s, H_mini={h_mini}, batch_size={batch_size}, expected_round={expected_round:.0}s",
            );
            // Persist node ID
            tokio::fs::create_dir_all(&cache_dir_expanded).await?;
            tokio::fs::write(&node_id_path, &node_id).await?;
            // Merge training params from coordinator
            if let Some(params) = reg.training_params {
                config.training_params = Some(params);
            }

            // GPU mode: use continuous training loop (GPU never idles)
            if !config.force_cpu {
                info!("Entering continuous GPU training mode");
                return distrain_node::continuous::run_continuous_training(
                    &mut config,
                    coordinator,
                    storage,
                    cache_dir,
                    node_id,
                    h_mini,
                    batch_size,
                    grad_accum_steps,
                    &manifest,
                    data_cache,
                    total_shards,
                    shards_per_node,
                    error_buffer,
                )
                .await;
            }
        }

        // CPU fallback: original round-based loop continues below

        // Compute deterministic shard assignment for this node + version
        let shard_ids = distrain_model::compute_shard_assignment(
            &node_id, version, total_shards, shards_per_node,
        );
        info!(
            "Shard assignment for v{version}: {} shards (e.g. {:?}...)",
            shard_ids.len(),
            &shard_ids[..shard_ids.len().min(5)],
        );

        // Resolve shard filenames from manifest indices
        let shard_names: Vec<String> = shard_ids
            .iter()
            .filter_map(|&idx| manifest.shards.get(idx).map(|e| e.filename.clone()))
            .collect();

        // Streaming data loader: downloads first few shards, starts training immediately.
        // Remaining shards downloaded on demand as training consumes data.
        let max_loaded_shards = 5;
        let mut streaming_loader = data::StreamingDataLoader::new(
            storage.clone(),
            shard_names,
            data_cache.clone(),
            config.seq_len,
            max_loaded_shards,
        ).await?;

        // Seek to a different position each round so re-training the same checkpoint
        // version doesn't produce identical batches.
        streaming_loader.seek_by_seed(seq_num);
        info!(
            "Data loaded (streaming): {} tokens from {}/{} shards (rest on demand)",
            streaming_loader.total_tokens_available(),
            streaming_loader.shards_loaded(),
            streaming_loader.total_shards(),
        );
        resources::log_memory("after shard loading");

        // No mid-round checkpoint polling. Train the full round, push, then check
        // for new checkpoints at the top of the loop. Staleness weighting handles
        // the case where the checkpoint advanced during training.

        // Train H_mini steps
        seq_num += 1;
        let delta_path = cache_dir.join(format!("delta_{node_id}_{seq_num}.delta.zst"));

        // Create a DataLoader snapshot from the streaming loader's currently loaded shards.
        // The trainer functions expect &mut DataLoader, so we bridge here.
        let mut data_loader = streaming_loader.to_data_loader(batch_size)?;

        // GPU path: use watchdog. CPU path: direct call (no driver hang risk).
        let max_mem_fraction = config.max_memory_fraction;
        let result = if !config.force_cpu {
            // GPU training with watchdog (timeout + catch_unwind)
            // force_batch_size without calibration: use generous default (no hang risk, just slow first round)
            let sps = gpu_secs_per_step.unwrap_or(if user_batch_size.is_some() { 300.0 } else { 10.0 });
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
                {
                    let hb_url = config.coordinator_url.clone();
                    let hb_node_id = node_id.clone();
                    let hb_version = version;
                    move |progress: trainer::StepProgress| {
                        // Memory pressure check every 50 steps
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
                        // Heartbeat every 10 steps, not every step.
                        // The sync HTTP call can block training if coordinator is slow.
                        if progress.step % 10 != 0 && progress.step + 1 != progress.total_steps {
                            return false;
                        }
                        let client = crate::client::CoordinatorClient::new(&hb_url);
                        match client.heartbeat_sync(
                            &hb_node_id,
                            Some(progress.step),
                            Some(progress.total_steps),
                            Some(progress.loss),
                            Some(hb_version),
                        ) {
                            Ok(resp) if resp.should_abort => {
                                warn!(
                                    "Coordinator says abort: checkpoint advanced to v{}",
                                    resp.latest_version.unwrap_or(0),
                                );
                                return true;
                            }
                            Err(e) => {
                                // Best-effort, don't abort on heartbeat failure
                                tracing::debug!("Heartbeat failed: {e}");
                            }
                            _ => {}
                        }
                        false
                    }
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
                    if user_batch_size.is_some() {
                        // force_batch_size means user trusts GPU; retry instead of falling back
                        warn!(
                            "GPU hung during training (timeout {timeout_secs:.0}s) — retrying on GPU (force_batch_size set)"
                        );
                        continue;
                    }
                    // No force_batch_size — refuse silent CPU fallback
                    error!(
                        "GPU hung during training (timeout {timeout_secs:.0}s). \
                         This is a fatal error — a GPU node should not silently fall back to CPU. \
                         Check your GPU driver and VRAM. Use --cpu to explicitly request CPU training."
                    );
                    return Err(anyhow::anyhow!("GPU training failed (hung, timeout {timeout_secs:.0}s), refusing silent CPU fallback"));
                }
                Err(trainer::TrainingFailure::GpuPanic { message }) => {
                    if user_batch_size.is_some() {
                        // OOM recovery with force_batch_size: halve batch_size, double grad_accum
                        if batch_size > 1 {
                            let old_bs = batch_size;
                            batch_size /= 2;
                            grad_accum_steps = effective_batch_size / batch_size;
                            warn!(
                                "GPU panic at batch_size={old_bs}: {message} — recovering to batch_size={batch_size}, grad_accum={grad_accum_steps}"
                            );
                            continue; // Retry round with smaller batch
                        }
                        // batch_size=1 and still failing with force_batch_size — fatal
                        error!(
                            "GPU panicked at batch_size=1 even with force_batch_size: {message}. \
                             This is a fatal error — check your GPU driver and VRAM. \
                             Use --cpu to explicitly request CPU training."
                        );
                        return Err(anyhow::anyhow!("GPU training failed (panic at batch_size=1), refusing silent CPU fallback"));
                    }
                    // OOM recovery without force_batch_size: try halving first
                    if batch_size > 1 {
                        let old_bs = batch_size;
                        batch_size /= 2;
                        grad_accum_steps = effective_batch_size / batch_size;
                        warn!(
                            "GPU panic at batch_size={old_bs}: {message} — recovering to batch_size={batch_size}, grad_accum={grad_accum_steps}"
                        );
                        continue; // Retry round with smaller batch
                    }
                    // batch_size=1 and still failing — refuse silent CPU fallback
                    error!(
                        "GPU panicked at batch_size=1: {message}. \
                         This is a fatal error — a GPU node should not silently fall back to CPU. \
                         Check your GPU driver and VRAM. Use --cpu to explicitly request CPU training."
                    );
                    return Err(anyhow::anyhow!("GPU training failed (panic at batch_size=1), refusing silent CPU fallback"));
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
                {
                    let hb_url = config.coordinator_url.clone();
                    let hb_node_id = node_id.clone();
                    let hb_version = version;
                    move |progress: trainer::StepProgress| {
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
                        // Per-step heartbeat (sync, best-effort)
                        let client = crate::client::CoordinatorClient::new(&hb_url);
                        match client.heartbeat_sync(
                            &hb_node_id,
                            Some(progress.step),
                            Some(progress.total_steps),
                            Some(progress.loss),
                            Some(hb_version),
                        ) {
                            Ok(resp) if resp.should_abort => {
                                warn!(
                                    "Coordinator says abort: checkpoint advanced to v{}",
                                    resp.latest_version.unwrap_or(0),
                                );
                                return true;
                            }
                            Err(e) => {
                                tracing::debug!("Heartbeat failed: {e}");
                            }
                            _ => {}
                        }
                        false
                    }
                },
            )
            .await;

            if mem_abort.load(std::sync::atomic::Ordering::SeqCst) {
                memory_pressure_abort = true;
            }
            r
        }?;

        last_trained_version = Some(version);
        _result_elapsed = result.elapsed_secs;

        // Save node state to disk (resume point after crash/restart)
        {
            let node_state = distrain_node::resume::NodeState {
                last_checkpoint_version: version,
                seq_num,
                shard_index: streaming_loader.current_shard_index(),
                shard_offset: streaming_loader.current_token_offset(),
                node_id: node_id.clone(),
                saved_at: chrono::Utc::now().to_rfc3339(),
            };
            if let Err(e) = node_state.save(&cache_dir) {
                warn!("Failed to save node state: {e}");
            }
            if let Err(e) = distrain_node::resume::save_error_buffer(&error_buffer, &cache_dir) {
                warn!("Failed to save error buffer: {e}");
            }
        }

        // Pipeline: evict consumed shards and download next ones while we handle the result.
        // This ensures fresh shard data is ready for the next round.
        if let Err(e) = streaming_loader.ensure_next_shard().await {
            warn!("Failed to stream next shard (non-fatal, will retry next round): {e:#}");
        }

        // Phase 2: skip warmup after first round (model is already warm)
        if let Some(ref mut tp) = config.training_params {
            if tp.warmup_fraction > 0.0 {
                info!("Skipping warmup for subsequent rounds (model is warm)");
                tp.warmup_fraction = 0.0;
            }
        }
        info!(
            "Training done: {}/{} steps, loss={:.4}, tokens={}, time={:.1}s, batch_size={}, grad_accum={}, effective_batch={}",
            result.steps_completed, h_mini, result.final_loss, result.tokens_processed, result.elapsed_secs,
            result.batch_size, grad_accum_steps, effective_batch_size,
        );

        // Refine H_mini from actual training speed (first round or after fallback)
        if result.steps_completed > 0 && result.elapsed_secs > 0.0 {
            let actual_sps = result.elapsed_secs / result.steps_completed as f64;
            let refined = ((target_interval / actual_sps) as u64)
                .clamp(min_h, max_h);
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

        // Don't push if loss spike detected (delta would be harmful)
        if result.loss_spiked {
            warn!("Loss spike detected — skipping push (delta would be harmful)");
            let _ = tokio::fs::remove_file(&delta_path).await;
            continue;
        }

        // Adaptive inner LR: observe loss stability, adjust for next round
        if result.loss_variance > 0.0 && result.final_loss > 0.0 {
            let cv = result.loss_variance.sqrt() / result.final_loss;
            if cv > 0.5 {
                effective_lr_max = (effective_lr_max * 0.8).max(base_lr_max * 0.1);
                info!("Inner LR reduced: cv={cv:.3}, new lr_max={effective_lr_max:.2e}");
            } else if cv < 0.1 && effective_lr_max < base_lr_max {
                effective_lr_max = (effective_lr_max * 1.1).min(base_lr_max);
                info!("Inner LR restored: cv={cv:.3}, new lr_max={effective_lr_max:.2e}");
            }
            // Apply for next round
            if let Some(ref mut tp) = config.training_params {
                tp.lr_max = effective_lr_max;
            }
        }

        // GPU pipeline: spawn upload+push in background, continue to next round immediately.
        // All data needed by the background task is cloned/moved here.
        let delta_key =
            distrain_shared::paths::delta_path(version, &node_id, seq_num);
        let push_body = distrain_shared::types::DeltaPush {
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

        // Build metrics entry now (captures current values before they change next round)
        let metrics_entry = serde_json::json!({
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
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

        let storage_bg = storage.clone();
        let coordinator_bg = coordinator.clone();
        let delta_path_bg = delta_path.clone();
        let metrics_path_bg = cache_dir.join("node_metrics.jsonl");
        let cache_dir_bg = cache_dir.clone();
        let bps_shared = measured_upload_bps_shared.clone();

        pending_upload = Some(tokio::spawn(async move {
            // Upload delta to R2 (retry with exponential backoff, measure bandwidth)
            let delta_file_size = tokio::fs::metadata(&delta_path_bg).await.map(|m| m.len()).unwrap_or(0);
            let upload_start = std::time::Instant::now();
            let mut upload_ok = false;
            for attempt in 1..=5u32 {
                match storage_bg.upload_from_file(&delta_key, &delta_path_bg).await {
                    Ok(()) => {
                        upload_ok = true;
                        let upload_secs = upload_start.elapsed().as_secs_f64();
                        if upload_secs > 0.1 && delta_file_size > 0 {
                            let bps = delta_file_size as f64 / upload_secs;
                            *bps_shared.lock().unwrap() = Some(bps);
                            info!("Upload: {:.1}MB in {upload_secs:.1}s ({:.1} MB/s)",
                                delta_file_size as f64 / 1e6, bps / 1e6);
                        }
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
                let _ = tokio::fs::remove_file(&delta_path_bg).await;
                return Ok(());
            }

            // Push metadata to coordinator (retry with exponential backoff)
            let mut push_ok = false;
            for attempt in 1..=5u32 {
                match coordinator_bg.push_delta(&push_body).await {
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
                warn!("Failed to push delta after 5 attempts, continuing");
            }

            // Append per-round metrics to local JSONL (best-effort, for paper analysis)
            {
                use tokio::io::AsyncWriteExt;
                if let Ok(line) = serde_json::to_string(&metrics_entry) {
                    if let Ok(mut f) = tokio::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&metrics_path_bg)
                        .await
                    {
                        let _ = f.write_all(format!("{line}\n").as_bytes()).await;
                    }
                }
            }

            // Cleanup delta file
            let _ = tokio::fs::remove_file(&delta_path_bg).await;

            // Housekeeping: keep only the 3 most recent checkpoints in cache
            cleanup_old_checkpoints(&cache_dir_bg, 3).await;

            Ok(())
        }));

        // Continue immediately to next round — upload proceeds in background
    }

    // Await any pending upload before exiting (unreachable in practice — loop never breaks)
    #[allow(unreachable_code)]
    {
        if let Some(handle) = pending_upload.take() {
            let _ = handle.await;
        }
        Ok(())
    }
}

/// Single-GPU baseline: train without coordinator, log per-step loss.
/// For paper comparison: same model, same data, same hyperparameters as distributed run.
async fn run_baseline(
    checkpoint_path: &str,
    data_dir: &str,
    total_steps: u64,
    output_path: &str,
    force_cpu: bool,
    batch_size: usize,
    seq_len: usize,
    lr_max: f64,
) -> Result<()> {
    use burn::grad_clipping::GradientClippingConfig;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
    use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
    use distrain_model::model::{DistrainTransformerModule, compute_lm_loss, precompute_rope_tables};
    use distrain_model::training::cosine_lr;
    use distrain_model::checkpoint::load_safetensors_map;
    use distrain_model::{CpuBackend, CpuDevice, GpuBackend, GpuDevice};

    let ckpt_path = std::path::Path::new(checkpoint_path);
    let model_config = trainer::infer_model_config(ckpt_path)?;
    info!("Baseline training: {} params, {} steps, batch_size={batch_size}, lr_max={lr_max:.2e}",
        model_config.param_count(), total_steps);

    // Load data shards from directory
    let mut shard_files: Vec<std::path::PathBuf> = Vec::new();
    let mut entries = tokio::fs::read_dir(data_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("shard_") && name.ends_with(".bin") {
            shard_files.push(entry.path());
        }
    }
    shard_files.sort();
    info!("Found {} shards in {data_dir}", shard_files.len());
    anyhow::ensure!(!shard_files.is_empty(), "No shard files found in {data_dir}");

    let mut data_loader = crate::data::DataLoader::from_files(&shard_files, seq_len, batch_size)?;
    info!("DataLoader ready: {} tokens", data_loader.total_tokens());

    let warmup_steps = ((total_steps as f64 * 0.2) as u64).max(2) as usize;
    let lr_min = 1e-6;

    let mut out_file = tokio::fs::OpenOptions::new()
        .create(true).write(true).truncate(true)
        .open(output_path).await?;

    let start_params = load_safetensors_map(ckpt_path)?;

    macro_rules! run_baseline_loop {
        ($Backend:ty, $device:expr) => {{
            let device = $device;
            let (rope_cos, rope_sin) = precompute_rope_tables::<$Backend>(
                model_config.head_dim(), model_config.max_seq_len, model_config.rope_theta, &device,
            );
            let module = DistrainTransformerModule::<$Backend>::new(&model_config, &device);
            let mut module = module.load_state_dict(&start_params, &device);
            let mut optim = AdamWConfig::new()
                .with_weight_decay(0.1)
                .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
                .init::<$Backend, DistrainTransformerModule<$Backend>>();

            let start = std::time::Instant::now();
            let mut total_tokens: u64 = 0;

            for step in 0..total_steps {
                let lr = cosine_lr(step as usize, warmup_steps, total_steps as usize, lr_max, lr_min);
                let tokens = data_loader.next_batch_sized(batch_size);
                let batch = Tensor::<$Backend, 2, Int>::from_data(
                    TensorData::new(tokens, [batch_size, seq_len]), &device,
                );
                let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
                let loss_val: f64 = loss.clone().into_scalar().elem();
                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &module);
                module = optim.step(lr, module, grads_params);

                total_tokens += (batch_size * seq_len) as u64;
                let elapsed = start.elapsed().as_secs_f64();

                info!("Baseline step {}/{total_steps}: loss={loss_val:.4}, lr={lr:.2e}, tokens={total_tokens}",
                    step + 1);

                use tokio::io::AsyncWriteExt;
                let entry = serde_json::json!({
                    "step": step + 1, "loss": loss_val, "lr": lr,
                    "tokens": total_tokens, "elapsed_secs": elapsed,
                });
                if let Ok(line) = serde_json::to_string(&entry) {
                    let _ = out_file.write_all(format!("{line}\n").as_bytes()).await;
                }
            }
            info!("Baseline complete: {total_steps} steps, {total_tokens} tokens, {:.1}s",
                start.elapsed().as_secs_f64());
        }};
    }

    if force_cpu {
        let device: CpuDevice = Default::default();
        run_baseline_loop!(CpuBackend, device);
    } else {
        run_baseline_loop!(GpuBackend, GpuDevice::default());
    }

    info!("Results written to {output_path}");
    Ok(())
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
    match coordinator.register(config, None, None).await {
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

    // Allocate the same extra memory that real training uses:
    // - start_params snapshot (for delta computation)
    // - error buffer (for compression)
    // This ensures the stress test measures realistic available VRAM.
    let _start_params_snapshot = start_params.clone(); // ~500MB for Tiny 125M
    let _error_buffer: std::collections::HashMap<String, Vec<f32>> = start_params
        .iter()
        .map(|(k, v)| (k.clone(), vec![0.0f32; v.len()]))
        .collect(); // another ~500MB

    let start = Instant::now();
    let _secs_total = 0.0f64;
    let num_steps = 3u64; // multiple steps to catch memory growth from caching

    for step in 0..num_steps {
        let tokens: Vec<i64> = (0..(batch_size * seq_len))
            .map(|i| (trainer::splitmix64_pub((i + step as usize * batch_size * seq_len) as u64) % model_config.vocab_size as u64) as i64)
            .collect();
        let batch = Tensor::<GpuBackend, 2, Int>::from_data(
            TensorData::new(tokens, [batch_size, seq_len]), &device,
        );
        let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
        let loss_val: f64 = loss.clone().into_scalar().elem();

        if loss_val == 0.0 || loss_val.is_nan() || loss_val.is_infinite() || loss_val < 0.0 {
            anyhow::bail!("GPU produced invalid loss={loss_val} at step {step}");
        }

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optim.step(3e-4, module, grads_params);
    }

    let secs_per_step = start.elapsed().as_secs_f64() / num_steps as f64;
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
        config.seq_len, config.batch_size.unwrap_or(4),
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
                TensorData::new(tokens, [config.batch_size.unwrap_or(4), config.seq_len]),
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
