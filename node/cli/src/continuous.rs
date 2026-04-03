//! Continuous training loop — GPU never idles between rounds.
//!
//! Instead of the stop-download-restart cycle of the original loop, continuous
//! training keeps the model and optimizer alive on the GPU. Three concurrent
//! tasks communicate via channels:
//!
//! 1. **checkpoint_manager** (async) — polls coordinator for new checkpoints,
//!    downloads safetensors, loads state_dict, sends to training thread.
//! 2. **delta_uploader** (async) — receives compressed deltas from training
//!    thread, uploads to R2, pushes metadata to coordinator.
//! 3. **continuous_training_loop** (blocking, spawn_blocking) — tight
//!    forward/backward/optim loop. Checks for new checkpoints between steps
//!    via try_recv. Queues deltas for upload every H_mini steps.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use tracing::{info, warn, error};

use distrain_model::checkpoint::load_safetensors_map;
use distrain_model::compression::{
    compress_delta, CompressionConfig, CompressionStats, ErrorBuffer,
};
use distrain_model::config::ModelConfig;
use distrain_model::model::{
    compute_lm_loss, precompute_rope_tables, DistrainTransformerModule,
};
use distrain_model::training::compute_outer_delta;
use distrain_model::{GpuBackend, GpuDevice};
use distrain_shared::storage::Storage;
use distrain_shared::types::DeltaPush;

use crate::client::CoordinatorClient;
use crate::trainer::{adaptive_top_k, infer_model_config};

// ── Channel types ───────────────────────────────────────────────────────

/// Sent from checkpoint_manager to the training thread when a new checkpoint
/// is available.
pub struct CheckpointSignal {
    pub version: u64,
    pub state_dict: HashMap<String, Vec<f32>>,
}

/// Sent from the training thread to delta_uploader with everything needed
/// to upload and push a delta.
pub struct DeltaPackage {
    /// Raw uncompressed delta — compression happens in delta_uploader (async, off GPU thread)
    pub raw_delta: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub delta_key: String,
    pub push_body: DeltaPush,
    pub seq_num: u64,
    pub training_loss: f64,
    pub tokens_processed: u64,
    pub mean_loss: f64,
    pub bandwidth_bps: u64,
}

// ── Checkpoint Manager ──────────────────────────────────────────────────

/// Polls coordinator for new checkpoints and sends them to the training thread.
///
/// Runs as a tokio task. When a new version is detected, downloads the
/// safetensors file, loads it into a HashMap<String, Vec<f32>>, and sends
/// a CheckpointSignal through the channel.
async fn checkpoint_manager(
    coordinator: CoordinatorClient,
    storage: Storage,
    cache_dir: PathBuf,
    _node_id: String,
    checkpoint_tx: std::sync::mpsc::Sender<CheckpointSignal>,
    initial_version: u64,
) {
    let mut known_version = initial_version;
    let poll_interval = std::time::Duration::from_secs(5);

    loop {
        tokio::time::sleep(poll_interval).await;

        let ckpt_info = match coordinator.get_latest_checkpoint().await {
            Ok(info) => info,
            Err(e) => {
                warn!("Checkpoint poll failed (non-fatal): {e:#}");
                continue;
            }
        };

        if ckpt_info.version <= known_version {
            continue;
        }

        let new_version = ckpt_info.version;
        info!(
            "New checkpoint detected: v{known_version} -> v{new_version}, downloading..."
        );

        let ckpt_path = cache_dir.join(format!("v{new_version}_model.safetensors"));

        // Download with retries
        if !ckpt_path.exists() {
            let mut ok = false;
            for attempt in 1..=3 {
                match storage
                    .download_to_file(&ckpt_info.checkpoint_key, &ckpt_path)
                    .await
                {
                    Ok(_) => {
                        ok = true;
                        break;
                    }
                    Err(e) => {
                        warn!("Checkpoint download failed (attempt {attempt}/3): {e:#}");
                        let _ = tokio::fs::remove_file(&ckpt_path).await;
                        if attempt < 3 {
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        }
                    }
                }
            }
            if !ok {
                warn!("Failed to download v{new_version} after 3 attempts, will retry");
                continue;
            }
        }

        // Load state dict on a blocking thread (can be large for 7B)
        let path_clone = ckpt_path.clone();
        let state_dict = match tokio::task::spawn_blocking(move || {
            load_safetensors_map(&path_clone)
        })
        .await
        {
            Ok(Ok(sd)) => sd,
            Ok(Err(e)) => {
                warn!("Failed to load checkpoint v{new_version}: {e:#}");
                continue;
            }
            Err(e) => {
                warn!("Spawn_blocking panicked loading checkpoint: {e}");
                continue;
            }
        };

        info!(
            "Checkpoint v{new_version} loaded ({} tensors), sending to training thread",
            state_dict.len()
        );

        if checkpoint_tx
            .send(CheckpointSignal {
                version: new_version,
                state_dict,
            })
            .is_err()
        {
            info!("Training thread gone, checkpoint_manager exiting");
            return;
        }

        known_version = new_version;

        // Housekeeping: keep only recent checkpoints
        cleanup_old_checkpoints(&cache_dir, 3).await;
    }
}

// ── Delta Uploader ──────────────────────────────────────────────────────

/// Receives raw deltas from the training thread, compresses them (async, off GPU),
/// uploads to storage, and pushes metadata to the coordinator.
async fn delta_uploader(
    storage: Storage,
    coordinator: CoordinatorClient,
    mut delta_rx: tokio::sync::mpsc::Receiver<DeltaPackage>,
    cache_dir: PathBuf,
    measured_bandwidth_bps: Arc<AtomicU64>,
    recommended_h_mini: Arc<AtomicU64>,
) {
    let mut error_buffer = ErrorBuffer::new();

    while let Some(mut pkg) = delta_rx.recv().await {
        // Compress the raw delta (CPU work, runs async while GPU trains next round)
        let bw = pkg.bandwidth_bps;
        let upload_budget = if bw > 0 { bw * 10 } else { 0 };
        let raw_param_bytes: u64 = pkg.raw_delta.values().map(|v| v.len() as u64 * 4).sum();

        let compression_config = if bw == 0 {
            CompressionConfig {
                top_k_fraction: adaptive_top_k(pkg.mean_loss, None, 0),
                ..CompressionConfig::default()
            }
        } else {
            // Try raw first (top_k=1.0)
            let full_config = CompressionConfig {
                top_k_fraction: 1.0, quantize_int8: false, ..CompressionConfig::default()
            };
            let mut trial_eb = error_buffer.clone();
            if let Ok((trial, _)) = compress_delta(&pkg.raw_delta, &pkg.shapes, &full_config, &mut trial_eb) {
                if (trial.len() as u64) <= upload_budget {
                    info!("Bandwidth-adaptive: raw fits! {}MB (bw={:.1} MB/s)", trial.len() / (1024*1024), bw as f64 / 1e6);
                    full_config
                } else {
                    let k = adaptive_top_k(pkg.mean_loss, Some(upload_budget), raw_param_bytes);
                    info!("Bandwidth-adaptive: top-k {:.0}% (bw={:.1} MB/s)", k * 100.0, bw as f64 / 1e6);
                    CompressionConfig { top_k_fraction: k, quantize_int8: false, ..CompressionConfig::default() }
                }
            } else {
                let k = adaptive_top_k(pkg.mean_loss, Some(upload_budget), raw_param_bytes);
                CompressionConfig { top_k_fraction: k, quantize_int8: false, ..CompressionConfig::default() }
            }
        };

        let compressed_bytes = match compress_delta(&pkg.raw_delta, &pkg.shapes, &compression_config, &mut error_buffer) {
            Ok((compressed, stats)) => {
                info!("Compression: retention={:.1}%, {}MB → {}MB",
                    stats.retention_ratio * 100.0,
                    stats.raw_param_bytes / (1024*1024),
                    stats.compressed_bytes / (1024*1024));
                pkg.push_body.compressed_bytes = Some(stats.compressed_bytes);
                pkg.push_body.dense_norm = Some(stats.dense_norm);
                pkg.push_body.sparse_norm = Some(stats.sparse_norm);
                compressed
            }
            Err(e) => {
                warn!("Compression failed: {e:#}, skipping delta");
                continue;
            }
        };

        let delta_size = compressed_bytes.len();
        info!("Uploading delta seq={} ({:.1} MB)...", pkg.seq_num, delta_size as f64 / 1e6);

        // Upload compressed bytes to storage with retry
        let upload_start = Instant::now();
        let mut upload_ok = false;
        for attempt in 1..=5u32 {
            match storage.put(&pkg.delta_key, compressed_bytes.clone()).await {
                Ok(()) => {
                    let secs = upload_start.elapsed().as_secs_f64();
                    let bps = if secs > 0.0 { delta_size as f64 / secs } else { 0.0 };
                    info!(
                        "Upload: {:.1}MB in {secs:.1}s ({:.1} MB/s)",
                        delta_size as f64 / 1e6,
                        bps / 1e6
                    );

                    // Store measured bandwidth for adaptive compression
                    measured_bandwidth_bps.store(bps as u64, Ordering::Relaxed);

                    // Recommend H_mini based on bandwidth: target ~10s upload time.
                    // h_mini = steps that produce a delta uploadable in 10s.
                    // delta_bytes ≈ this upload's size. Steps = pkg.inner_steps.
                    // If upload took T seconds, then 10s would allow (10/T) × steps.
                    let target_upload_secs = 10.0f64;
                    let steps = pkg.push_body.inner_steps as f64;
                    let new_h = ((target_upload_secs / secs) * steps)
                        .clamp(10.0, 500.0) as u64;
                    recommended_h_mini.store(new_h, Ordering::Relaxed);

                    upload_ok = true;
                    break;
                }
                Err(e) => {
                    warn!("Delta upload failed (attempt {attempt}/5): {e:#}");
                    if attempt < 5 {
                        let backoff = std::time::Duration::from_secs(2u64.pow(attempt));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
        if !upload_ok {
            warn!("Failed to upload delta seq={} after 5 attempts, skipping", pkg.seq_num);
            continue;
        }

        // Push metadata to coordinator with retry
        for attempt in 1..=5u32 {
            match coordinator.push_delta(&pkg.push_body).await {
                Ok(resp) => {
                    if resp.accepted {
                        info!(
                            "Push accepted: seq={}, ckpt v{}",
                            pkg.seq_num, resp.checkpoint_version
                        );
                    } else {
                        info!(
                            "Push rejected: seq={}, reason={}",
                            pkg.seq_num,
                            resp.reason.unwrap_or_else(|| "unknown".into())
                        );
                    }
                    break;
                }
                Err(e) => {
                    warn!("Push failed (attempt {attempt}/5): {e:#}");
                    if attempt < 5 {
                        let backoff = std::time::Duration::from_secs(2u64.pow(attempt));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }

        // Append per-round metrics (best-effort)
        let metrics_entry = serde_json::json!({
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "seq": pkg.seq_num,
            "version": pkg.push_body.checkpoint_version,
            "steps": pkg.push_body.inner_steps,
            "loss": pkg.training_loss,
            "tokens": pkg.tokens_processed,
            "compressed_bytes": None::<CompressionStats>.as_ref().map(|s| s.compressed_bytes),
            "retention": None::<CompressionStats>.as_ref().map(|s| s.retention_ratio),
            "dense_norm": None::<CompressionStats>.as_ref().map(|s| s.dense_norm),
            "sparse_norm": None::<CompressionStats>.as_ref().map(|s| s.sparse_norm),
        });
        let metrics_path = cache_dir.join("node_metrics.jsonl");
        if let Ok(line) = serde_json::to_string(&metrics_entry) {
            use tokio::io::AsyncWriteExt;
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

    info!("Delta uploader: channel closed, exiting");
}

// ── Continuous Training Loop (blocking thread) ──────────────────────────

/// Parameters for the continuous training loop, gathered during setup.
pub struct ContinuousTrainingParams {
    pub checkpoint_path: PathBuf,
    pub model_config: ModelConfig,
    pub training_params: distrain_shared::types::TrainingParams,
    pub h_mini: u64,
    pub batch_size: usize,
    pub grad_accum_steps: usize,
    pub seq_len: usize,
    pub node_id: String,
    pub initial_version: u64,
    /// Starting seq_num (persists across inner loop restarts).
    pub initial_seq_num: u64,
    /// Pre-loaded batches for the first round.
    pub data_loader: crate::data::DataLoader,
    /// Use loss-based lr (adapts from training loss). Default true.
    /// Set LR_MODE=constant env var to disable.
    pub loss_based_lr: bool,
    /// Shared bandwidth measurement from delta_uploader (bytes/sec).
    pub measured_bandwidth_bps: Arc<AtomicU64>,
}

/// Tight training loop that runs on a blocking thread via spawn_blocking.
///
/// Creates the model and optimizer ONCE, then loops:
/// 1. Check for new checkpoint (try_recv) — if found, extract delta from
///    current partial round, queue upload, swap weights, decay error buffer.
/// 2. Every H_mini steps: extract delta, compress, queue upload, snapshot
///    current weights as new round_start.
/// 3. Train one step: forward, backward, optim.step().
/// 4. Heartbeat every 10 steps.
fn continuous_training_loop(
    params: ContinuousTrainingParams,
    checkpoint_rx: std::sync::mpsc::Receiver<CheckpointSignal>,
    delta_tx: tokio::sync::mpsc::Sender<DeltaPackage>,


    heartbeat_url: String,
) -> (crate::data::DataLoader, u64) {
    let device: GpuDevice = Default::default();
    let mut seq_num: u64 = params.initial_seq_num;
    let mut data_loader = params.data_loader;

    // Load initial checkpoint
    let start_params = match load_safetensors_map(&params.checkpoint_path) {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to load initial checkpoint: {e:#}");
            return (data_loader, seq_num);
        }
    };

    let (rope_cos, rope_sin) = precompute_rope_tables::<GpuBackend>(
        params.model_config.head_dim(),
        params.model_config.max_seq_len,
        params.model_config.rope_theta,
        &device,
    );

    let module = DistrainTransformerModule::<GpuBackend>::new(&params.model_config, &device);
    let mut module = module.load_state_dict(&start_params, &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(params.training_params.weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            params.training_params.grad_clip_norm as f32,
        )))
        .init::<GpuBackend, DistrainTransformerModule<GpuBackend>>();

    let micro_batch_size = params.batch_size;
    let grad_accum_steps = params.grad_accum_steps;

    // Scale base learning rate linearly with batch size (reference: batch=4).
    let reference_batch = 4usize;
    let effective_batch = micro_batch_size * grad_accum_steps;
    let lr_scale = effective_batch as f64 / reference_batch as f64;
    let lr_max = params.training_params.lr_max * lr_scale;
    info!("LR scaled {lr_scale:.1}x for batch={effective_batch}: lr_max={lr_max:.2e}");


    let h_mini = params.h_mini;
    let seq_len = params.seq_len;

    let effective_batch = micro_batch_size * grad_accum_steps;

    // Track state across rounds
    let current_version = params.initial_version;
    let mut global_step: u64 = 0;
    let mut round_step: u64 = 0;
    let mut round_start_params = start_params;
    let mut round_start_time = Instant::now();

    // Loss tracking
    let mut loss_sum: f64 = 0.0;
    let mut loss_count: u64 = 0;
    let mut loss_ema: f64 = 0.0;
    let mut last_loss: f64 = 0.0;
    let mut round_tokens: u64 = 0;

    // Warmup for first round only
    let warmup_steps = ((h_mini as f64 * params.training_params.warmup_fraction) as u64).max(2);
    let mut first_round = true;

    // Batch iterator — refilled on checkpoint swap

    info!(
        "Continuous training started: v{current_version}, H_mini={h_mini}, batch={micro_batch_size}x{grad_accum_steps}, lr={lr_max:.2e}"
    );

    loop {
        // ── 1. Check for new checkpoint (non-blocking) ──────────────
        match checkpoint_rx.try_recv() {
            Ok(signal) => {
                let swap_start = Instant::now();
                let new_version = signal.version;
                info!(
                    "Checkpoint swap: v{current_version} -> v{new_version} (mid-round at step {round_step}/{h_mini})"
                );

                // If we trained any steps this round, extract partial delta and queue upload
                if round_step > 0 && loss_count > 0 {
                    let mean_loss = loss_sum / loss_count as f64;
                    let bw = params.measured_bandwidth_bps.load(Ordering::Relaxed);
                    if let Some(pkg) = build_delta_package(
                        &module,
                        &round_start_params,
                        mean_loss,
                        current_version,
                        &params.node_id,
                        &mut seq_num,
                        round_step,
                        round_tokens,
                        round_start_time.elapsed().as_secs_f64(),
                        bw,
                    ) {
                        // Best-effort send (don't block training)
                        if delta_tx.blocking_send(pkg).is_err() {
                            warn!("Delta channel closed during checkpoint swap");
                            return (data_loader, seq_num);
                        }
                    }
                }

                // Keep error buffer on checkpoint change (100% retention).
                // The accumulated error is valid gradient signal from prior
                // training rounds — decaying it loses information permanently.
                info!("Keeping error buffer (100% retention) across checkpoint change");

                // We need new batches for the new checkpoint version (different shard
                // assignment). Return error_buffer so the outer loop in
                // run_continuous_training can re-invoke us with fresh data.
                info!(
                    "Checkpoint swap v{current_version} -> v{new_version} in {:.0}ms, returning for data refill",
                    swap_start.elapsed().as_millis()
                );
                return (data_loader, seq_num);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // No new checkpoint, continue training
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                info!("Checkpoint channel disconnected, training loop exiting");
                // Flush any partial round before exit
                if round_step > 0 && loss_count > 0 {
                    let mean_loss = loss_sum / loss_count as f64;
                    let bw = params.measured_bandwidth_bps.load(Ordering::Relaxed);
                    if let Some(pkg) = build_delta_package(
                        &module,
                        &round_start_params,
                        mean_loss,
                        current_version,
                        &params.node_id,
                        &mut seq_num,
                        round_step,
                        round_tokens,
                        round_start_time.elapsed().as_secs_f64(),
                        bw,
                    ) {
                        let _ = delta_tx.blocking_send(pkg);
                    }
                }
                return (data_loader, seq_num);
            }
        }

        // ── 2. Check if we've completed H_mini steps (push delta) ───
        if round_step >= h_mini && round_step > 0 {
            let mean_loss = if loss_count > 0 {
                loss_sum / loss_count as f64
            } else {
                last_loss
            };
            let elapsed = round_start_time.elapsed().as_secs_f64();

            info!(
                "Round complete: {round_step}/{h_mini} steps, mean_loss={mean_loss:.4}, tokens={round_tokens}, time={elapsed:.1}s"
            );

            // Don't push if only warmup steps completed
            let skip_push = first_round && round_step <= warmup_steps;
            if skip_push {
                warn!(
                    "First round had only {round_step} steps (warmup={warmup_steps}) — skipping push"
                );
            } else if let Some(pkg) = {
                let bw = params.measured_bandwidth_bps.load(Ordering::Relaxed);
                build_delta_package(
                    &module,
                    &round_start_params,
                    mean_loss,
                    current_version,
                    &params.node_id,
                    &mut seq_num,
                    round_step,
                    round_tokens,
                    elapsed,
                    bw,
                )
            } {
                if delta_tx.blocking_send(pkg).is_err() {
                    warn!("Delta channel closed, training loop exiting");
                    return (data_loader, seq_num);
                }
            }

            // Snapshot current params as new round start
            round_start_params = module.extract_state_dict();
            round_step = 0;
            loss_sum = 0.0;
            loss_count = 0;
            round_tokens = 0;
            round_start_time = Instant::now();
            first_round = false;
        }

        // ── 3. Get next batch (on the fly, never repeats) ──────────
        let tokens = data_loader.next_batch_sized(micro_batch_size);

        // ── 4. Train one step ───────────────────────────────────────
        // LR mode: constant (default, matches baseline) or loss-based (adapts from loss_ema).
        // Set LR_MODE=loss_based env var to enable adaptive lr.
        let lr = if first_round && round_step < warmup_steps {
            lr_max * (round_step as f64 / warmup_steps as f64)
        } else if params.loss_based_lr && loss_ema > 0.0 {
            let ln_vocab = (params.model_config.vocab_size as f64).ln();
            let ratio = (loss_ema / ln_vocab).clamp(0.1, 1.0);
            lr_max * ratio
        } else {
            lr_max
        };

        // Forward pass
        let batch = Tensor::<GpuBackend, 2, Int>::from_data(
            TensorData::new(tokens, [micro_batch_size, seq_len]),
            &device,
        );

        let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
        let loss_val: f64 = loss.clone().into_scalar().elem();

        if loss_val.is_nan() {
            warn!("Step {global_step}: NaN loss, skipping update");
            continue;
        }

        // Spike detection
        if loss_count > 3 {
            if loss_val > loss_ema * 3.0 && loss_val > loss_ema + 5.0 {
                warn!(
                    "Loss spike: step {global_step} loss={loss_val:.1} vs ema={loss_ema:.1}"
                );
                // Don't abort — just skip this step's update
                continue;
            }
        }

        // Track loss (only on first micro-batch of each step)
        last_loss = loss_val;
        loss_sum += loss_val;
        loss_count += 1;
        loss_ema = if loss_count == 1 {
            loss_val
        } else {
            loss_ema * 0.9 + loss_val * 0.1
        };

        // Backward + optimizer step
        let scaled_loss = loss / (grad_accum_steps as f32);
        let grads = scaled_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optim.step(lr / grad_accum_steps as f64, module, grads_params);

        round_step += 1;
        global_step += 1;
        round_tokens += (effective_batch * seq_len) as u64;

        if round_step % 10 == 0 || round_step == 1 {
            info!(
                "Step {round_step}/{h_mini} (global {global_step}): loss={loss_val:.4}, lr={lr:.2e}"
            );
        }

        // ── 5. Heartbeat every 10 steps ─────────────────────────────
        if global_step % 10 == 0 {
            let client = CoordinatorClient::new(&heartbeat_url);
            match client.heartbeat_sync(
                &params.node_id,
                Some(round_step),
                Some(h_mini),
                Some(loss_val),
                Some(current_version),
            ) {
                Ok(resp) if resp.should_abort => {
                    // Coordinator says abort — a new checkpoint exists.
                    // Don't abort immediately; the checkpoint_manager will
                    // deliver the new checkpoint soon via the channel.
                    info!(
                        "Coordinator suggests abort (v{} available), waiting for checkpoint_manager",
                        resp.latest_version.unwrap_or(0)
                    );
                }
                Err(e) => {
                    tracing::debug!("Heartbeat failed: {e}");
                }
                _ => {}
            }
        }
    }
}

/// Snapshot delta from current model vs round start. No compression here —
/// compression happens async in delta_uploader so GPU can start next round immediately.
fn build_delta_package(
    module: &DistrainTransformerModule<GpuBackend>,
    round_start_params: &HashMap<String, Vec<f32>>,
    mean_loss: f64,
    version: u64,
    node_id: &str,
    seq_num: &mut u64,
    steps: u64,
    tokens: u64,
    elapsed_secs: f64,
    bandwidth_bps: u64,
) -> Option<DeltaPackage> {
    let current_params = module.extract_state_dict();
    let shapes = module.extract_shapes();
    let delta = compute_outer_delta(round_start_params, &current_params);

    *seq_num += 1;
    let delta_key = distrain_shared::paths::delta_path(version, node_id, *seq_num);

    let push_body = DeltaPush {
        node_id: distrain_shared::types::NodeId(node_id.to_string()),
        seq_num: *seq_num,
        checkpoint_version: version,
        inner_steps: steps,
        delta_key: delta_key.clone(),
        training_loss: mean_loss,
        tokens_processed: tokens,
        training_time_secs: elapsed_secs,
        compressed_bytes: None,
        dense_norm: None,
        sparse_norm: None,
    };

    info!("Delta snapshot ready ({}MB raw), sending to uploader for async compression",
        delta.values().map(|v| v.len() * 4).sum::<usize>() / (1024 * 1024));

    Some(DeltaPackage {
        raw_delta: delta,
        shapes,
        delta_key,
        push_body,
        seq_num: *seq_num,
        training_loss: mean_loss,
        tokens_processed: tokens,
        mean_loss,
        bandwidth_bps,
    })
}

// ── Main Entry Point ────────────────────────────────────────────────────

/// Run continuous GPU training — replaces the round-based loop in main.rs.
///
/// Call this after the setup phase (GPU probe, calibration, registration).
/// This function never returns under normal operation.
pub async fn run_continuous_training(
    config: &mut distrain_shared::config::NodeConfig,
    coordinator: CoordinatorClient,
    storage: Storage,
    cache_dir: PathBuf,
    node_id: String,
    mut h_mini: u64,
    mut batch_size: usize,
    grad_accum_steps: usize,
    manifest: &crate::data::Manifest,
    data_cache: PathBuf,
    total_shards: usize,
    shards_per_node: usize,
    error_buffer: ErrorBuffer,
) -> Result<()> {
    // Persistent state across restarts of the inner loop
    let mut seq_num: u64 = 0;
    let mut streaming_loader: Option<crate::data::StreamingDataLoader> = None;
    let mut data_loader: Option<crate::data::DataLoader> = None;

    // Shared atomics for bandwidth-adaptive compression and H_mini auto-tune
    let measured_bandwidth_bps = Arc::new(AtomicU64::new(0));
    let recommended_h_mini = Arc::new(AtomicU64::new(0));

    loop {
        // Get latest checkpoint
        let ckpt_info = coordinator.get_latest_checkpoint().await?;
        let version = ckpt_info.version;
        info!("Continuous training: starting from v{version}");

        // Download checkpoint if not cached
        let ckpt_path = cache_dir.join(format!("v{version}_model.safetensors"));
        if !ckpt_path.exists() {
            let mut ok = false;
            for attempt in 1..=3 {
                info!("Downloading checkpoint v{version} (attempt {attempt}/3)...");
                match storage
                    .download_to_file(&ckpt_info.checkpoint_key, &ckpt_path)
                    .await
                {
                    Ok(_) => {
                        ok = true;
                        break;
                    }
                    Err(e) => {
                        warn!("Download failed: {e:#}");
                        let _ = tokio::fs::remove_file(&ckpt_path).await;
                        if attempt < 3 {
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        }
                    }
                }
            }
            if !ok {
                warn!("Failed to download v{version}, retrying...");
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                continue;
            }
        }

        let model_config = infer_model_config(&ckpt_path)?;
        let seq_len = config.seq_len;

        // Shard assignment: computed once and reused across checkpoint changes.
        // Diversity comes from multiple nodes having different assignments, not from
        // reshuffling one node's shards every checkpoint (which wastes minutes reloading).
        // Data loader is only created on first iteration; subsequent checkpoints reuse it.
        if streaming_loader.is_none() {
            let shard_ids = distrain_model::compute_shard_assignment(
                &node_id, 0, total_shards, shards_per_node,
            );
            info!("Shard assignment: {} shards (fixed across checkpoints)", shard_ids.len());

            let shard_names: Vec<String> = shard_ids
                .iter()
                .filter_map(|&idx| manifest.shards.get(idx).map(|e| e.filename.clone()))
                .collect();

            let max_loaded_shards = {
                let budget = crate::resources::compute_memory_budget(&model_config, config);
                match budget {
                    Ok(b) => b.max_shards.max(2).min(shard_ids.len()),
                    Err(_) => 5,
                }
            };
            info!("Streaming data loader: max_loaded_shards={max_loaded_shards} (memory-based)");
            streaming_loader = Some(crate::data::StreamingDataLoader::new(
                storage.clone(),
                shard_names,
                data_cache.clone(),
                seq_len,
                max_loaded_shards,
            ).await?);
        }
        let loader = streaming_loader.as_mut().unwrap();
        // DataLoader created once, persists across checkpoint changes.
        // Walks sequentially through 2.2B tokens, never resets to position 0.
        if data_loader.is_none() {
            data_loader = Some(loader.to_data_loader(batch_size)?);
            info!("Data loader ready ({} tokens available)", loader.total_tokens_available());
        }

        // Set up channels
        // std::sync::mpsc: checkpoint_manager (async) -> training thread (blocking)
        let (checkpoint_tx, checkpoint_rx) = std::sync::mpsc::channel::<CheckpointSignal>();
        // tokio::sync::mpsc: training thread (blocking) -> delta_uploader (async)
        let (delta_tx, delta_rx) = tokio::sync::mpsc::channel::<DeltaPackage>(8);

        // Spawn checkpoint manager
        let cm_coordinator = coordinator.clone();
        let cm_storage = storage.clone();
        let cm_cache_dir = cache_dir.clone();
        let cm_node_id = node_id.clone();
        let cm_handle = tokio::spawn(checkpoint_manager(
            cm_coordinator,
            cm_storage,
            cm_cache_dir,
            cm_node_id,
            checkpoint_tx,
            version,
        ));

        // Spawn delta uploader
        let du_storage = storage.clone();
        let du_coordinator = coordinator.clone();
        let du_cache_dir = cache_dir.clone();
        let du_bw = measured_bandwidth_bps.clone();
        let du_h_mini = recommended_h_mini.clone();
        let du_handle = tokio::spawn(delta_uploader(
            du_storage,
            du_coordinator,
            delta_rx,
            du_cache_dir,
            du_bw,
            du_h_mini,
        ));

        // Build training params
        let training_params = config
            .training_params
            .as_ref()
            .cloned()
            .unwrap_or_default();

        let cont_params = ContinuousTrainingParams {
            checkpoint_path: ckpt_path,
            model_config,
            training_params,
            h_mini,
            batch_size,
            grad_accum_steps,
            seq_len,
            node_id: node_id.clone(),
            initial_version: version,
            initial_seq_num: seq_num,
            data_loader: data_loader.take().unwrap(),
            loss_based_lr: std::env::var("LR_MODE").map_or(true, |v| v != "constant"),
            measured_bandwidth_bps: measured_bandwidth_bps.clone(),
        };

        let heartbeat_url = config.coordinator_url.clone();

        // Run training on blocking thread (GPU). Compression happens in delta_uploader (async).
        let result = tokio::task::spawn_blocking(move || {
            continuous_training_loop(
                cont_params,
                checkpoint_rx,
                delta_tx,
                heartbeat_url,
            )
        })
        .await;

        let panicked = result.is_err();
        let (dl_out, new_seq_num) = result.unwrap_or_else(|e| {
            error!("Training thread panicked: {e}");
            (data_loader.take().unwrap_or_else(|| {
                // Shouldn't happen, but create a dummy
                crate::data::DataLoader::from_tokens(vec![vec![0u16; 1024]], 512, 4).unwrap()
            }), seq_num)
        });

        data_loader = Some(dl_out);
        seq_num = new_seq_num;

        // If training panicked (likely GPU OOM), reduce batch_size and retry.
        // The inner loop creates model/optimizer fresh each iteration, so reducing
        // batch_size here means the next iteration uses less GPU memory.
        // cubecl may leak some memory after panic, but reducing batch_size compensates.
        if panicked {
            let old_bs = batch_size;
            batch_size = (batch_size / 2).max(1);
            if batch_size == old_bs {
                // Already at 1 and still failing — GPU is broken, exit for restart
                error!("GPU OOM at batch_size=1. Exiting for clean restart.");
                std::process::exit(1);
            }
            warn!("Training panicked — reducing batch_size: {old_bs} → {batch_size}");
            // Brief pause to let GPU memory settle
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }

        // H_mini auto-tune based on measured bandwidth, clamped to config bounds
        let recommended = recommended_h_mini.load(Ordering::Relaxed);
        if recommended > 0 {
            let clamped = recommended.clamp(config.min_inner_steps, config.max_inner_steps);
            if clamped != h_mini {
                info!("Bandwidth-based H_mini: {h_mini} -> {clamped} (raw={recommended}, bounds=[{},{}])",
                    config.min_inner_steps, config.max_inner_steps);
                h_mini = clamped;
            }
        }

        // Clean up tasks
        cm_handle.abort();
        du_handle.abort();

        // Skip warmup for subsequent rounds
        if let Some(ref mut tp) = config.training_params {
            if tp.warmup_fraction > 0.0 {
                info!("Skipping warmup for subsequent rounds");
                tp.warmup_fraction = 0.0;
            }
        }

        info!("Training loop returned, restarting with latest checkpoint...");
    }
}

// ── Housekeeping ────────────────────────────────────────────────────────

/// Delete old cached checkpoints, keeping only the N most recent.
async fn cleanup_old_checkpoints(cache_dir: &Path, keep: usize) {
    let mut ckpts: Vec<(u64, PathBuf)> = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(cache_dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('v') && name.ends_with("_model.safetensors") {
                if let Some(ver_str) = name
                    .strip_prefix('v')
                    .and_then(|s| s.strip_suffix("_model.safetensors"))
                {
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
