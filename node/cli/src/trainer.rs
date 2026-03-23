//! Native Burn training engine.
//!
//! Uses wgpu backend (Metal on macOS, Vulkan on Linux/Windows) for GPU
//! acceleration. Falls back descriptions to CPU (ndarray) for calibration
//! and environments without GPU.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::ElementConversion;
use burn::tensor::{Int, Tensor, TensorData};
use serde::Deserialize;
use tracing::{info, warn};

use distrain_model::checkpoint::{load_safetensors_map, load_safetensors_shapes};
use distrain_model::compression::{CompressionConfig, ErrorBuffer, compress_delta, CompressionStats};
use distrain_model::config::{ModelConfig, ModelPreset};
use distrain_model::model::{
    compute_lm_loss, precompute_rope_tables, DistrainTransformerModule,
};
use distrain_model::training::{compute_outer_delta, cosine_lr};
use distrain_model::{CpuBackend, CpuDevice, GpuBackend, GpuDevice};

/// Adaptive top-k: increase sparsification fraction as loss decreases.
/// Early training has big gradients concentrated in few params — 1% captures signal.
/// Late training has small gradients spread across many params — need more.
fn adaptive_top_k(loss: f64) -> f32 {
    // Phase 2: 10x higher top-k. 250MB per push is fine on residential internet.
    let k = if loss > 20.0 {
        0.05
    } else if loss > 10.0 {
        0.10
    } else if loss > 7.0 {
        0.15
    } else if loss > 5.0 {
        0.20
    } else if loss > 3.0 {
        0.25
    } else {
        0.30
    };
    info!("Adaptive top-k: loss={loss:.2} → k={k} ({:.1}%)", k * 100.0);
    k
}

#[derive(Debug, Deserialize)]
pub struct TrainingResult {
    pub final_loss: f64,
    pub tokens_processed: u64,
    pub elapsed_secs: f64,
    pub steps_completed: u64,
    pub batch_size: usize,
    #[serde(default)]
    pub compression_stats: Option<CompressionStats>,
    #[serde(default)]
    pub loss_spiked: bool,
    #[serde(default)]
    pub loss_variance: f64,
}

/// Why a GPU training attempt failed.
#[derive(Debug)]
pub enum TrainingFailure {
    /// GPU hung and did not respond within the timeout.
    GpuHung { timeout_secs: f64 },
    /// GPU panicked (shader crash, OOM, driver error).
    GpuPanic { message: String },
    /// Non-GPU error (IO, data, etc.).
    Other(anyhow::Error),
}

impl std::fmt::Display for TrainingFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuHung { timeout_secs } => {
                write!(f, "GPU hung (timeout after {timeout_secs:.0}s)")
            }
            Self::GpuPanic { message } => write!(f, "GPU panic: {message}"),
            Self::Other(e) => write!(f, "{e:#}"),
        }
    }
}

impl From<anyhow::Error> for TrainingFailure {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

/// Progress info emitted during training.
#[derive(Debug, Clone)]
pub struct StepProgress {
    pub step: u64,
    pub total_steps: u64,
    pub loss: f64,
    pub lr: f64,
    pub tokens_processed: u64,
    pub elapsed_secs: f64,
}

/// Result of probing GPU hardware.
#[derive(Debug, Clone)]
pub enum GpuVerdict {
    /// GPU adapter found with details.
    Available {
        name: String,
        backend: String,
        is_integrated: bool,
        max_buffer_size: u64,
    },
    /// No GPU adapter found at all — use CPU.
    NoAdapter,
}

/// Probe for a GPU adapter. Does NOT run any compute — just checks if an
/// adapter exists and queries its capabilities.
pub async fn probe_gpu() -> GpuVerdict {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await;

    match adapter {
        Ok(a) => {
            let gpu_info = a.get_info();
            let name = gpu_info.name.clone();
            let backend = format!("{:?}", gpu_info.backend);
            let is_integrated = gpu_info.device_type == wgpu::DeviceType::IntegratedGpu;
            let limits = a.limits();
            let max_buffer_size = limits.max_buffer_size;
            info!(
                "GPU detected: {name} (vendor={}, type={:?}, backend={backend}, max_buffer={}MB)",
                gpu_info.vendor, gpu_info.device_type, max_buffer_size / (1024 * 1024),
            );
            GpuVerdict::Available { name, backend, is_integrated, max_buffer_size }
        }
        Err(e) => {
            info!("No GPU adapter found: {e}");
            GpuVerdict::NoAdapter
        }
    }
}

/// Run device calibration.
///
/// Uses the MicroTest config (64h/2L) which is fast on any device.
/// The calibration measures device speed to determine H_mini.
/// Returns (h_mini, use_cpu).
///
/// GPU validation strategy (no platform/vendor hardcoding):
/// 1. No adapter? → CPU.
/// 2. Adapter found? → Run a real forward+backward on GPU inside catch_unwind.
/// 3. If GPU calibration panics (shader compilation, OOM) → CPU.
/// 4. If GPU calibration produces bad results (loss=0, NaN, negative) → CPU.
/// 5. Otherwise → GPU works, use it.
pub async fn calibrate(config: &distrain_shared::config::NodeConfig) -> Result<(u64, bool)> {
    if config.force_cpu {
        let h = calibrate_cpu(config).await?;
        return Ok((h, true));
    }

    let verdict = probe_gpu().await;
    let gpu_name = match verdict {
        GpuVerdict::NoAdapter => {
            info!("No GPU adapter — calibrating on CPU");
            let h = calibrate_cpu(config).await?;
            return Ok((h, true));
        }
        GpuVerdict::Available { name, .. } => name,
    };

    // GPU adapter exists — try running actual compute on it.
    // Wrap in catch_unwind + timeout to handle panics AND hangs.
    // Some GPUs (e.g. Intel Iris Plus) lock up entirely when Metal shaders fail,
    // so catch_unwind alone isn't enough — we need a hard timeout.
    let config_clone = config.clone();
    let gpu_task = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build tokio runtime for GPU calibration");
            rt.block_on(calibrate_gpu_validated(&config_clone))
        }))
    });

    // 60s timeout — MicroTest calibration (5 steps) should finish well within this.
    // If the GPU hangs (shader lockup, driver deadlock), we bail.
    let gpu_result = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        gpu_task,
    ).await;

    match gpu_result {
        Ok(Ok(Ok(Ok(h)))) => {
            info!("GPU calibration succeeded on '{gpu_name}': H_mini = {h}");
            Ok((h, false))
        }
        Ok(Ok(Ok(Err(e)))) => {
            warn!("GPU calibration failed on '{gpu_name}': {e:#} — falling back to CPU");
            let h = calibrate_cpu(config).await?;
            Ok((h, true))
        }
        Ok(Ok(Err(panic_info))) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            warn!("GPU calibration panicked on '{gpu_name}': {msg} — falling back to CPU");
            let h = calibrate_cpu(config).await?;
            Ok((h, true))
        }
        Ok(Err(join_err)) => {
            warn!("GPU calibration task failed on '{gpu_name}': {join_err} — falling back to CPU");
            let h = calibrate_cpu(config).await?;
            Ok((h, true))
        }
        Err(_) => {
            warn!("GPU calibration timed out on '{gpu_name}' (60s) — GPU is hung, falling back to CPU");
            let h = calibrate_cpu(config).await?;
            Ok((h, true))
        }
    }
}

pub async fn calibrate_cpu(config: &distrain_shared::config::NodeConfig) -> Result<u64> {
    let target_interval = config.target_push_interval_secs;
    let min_h = config.min_inner_steps;
    let max_h = config.max_inner_steps;

    let device: CpuDevice = Default::default();
    let model_config = ModelPreset::MicroTest.config();
    let (rope_cos, rope_sin) = precompute_rope_tables::<CpuBackend>(
        model_config.head_dim(),
        model_config.max_seq_len,
        model_config.rope_theta,
        &device,
    );

    let mut module = DistrainTransformerModule::<CpuBackend>::new(&model_config, &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<CpuBackend, DistrainTransformerModule<CpuBackend>>();

    let calibration_steps = 5u64;
    let seq_len = 32usize;
    let batch_size = 2usize;
    let start = Instant::now();

    for step in 0..calibration_steps {
        let tokens: Vec<i64> = (0..(batch_size * seq_len))
            .map(|i| {
                (splitmix64(step * 1000 + i as u64) % model_config.vocab_size as u64) as i64
            })
            .collect();
        let batch = Tensor::<CpuBackend, 2, Int>::from_data(
            TensorData::new(tokens, [batch_size, seq_len]),
            &device,
        );

        let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optim.step(3e-4, module, grads_params);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let secs_per_step = elapsed / calibration_steps as f64;
    let h_mini = ((target_interval / secs_per_step) as u64).clamp(min_h, max_h);

    info!("Calibration (CPU): {secs_per_step:.3}s/step → H_mini = {h_mini}");
    Ok(h_mini)
}

/// GPU calibration with result validation.
///
/// Runs a real forward+backward pass and validates that the GPU produces
/// sane loss values. If loss is 0, NaN, negative, or infinite, the GPU is
/// producing garbage (broken shaders, incompatible hardware) and we bail
/// so the caller can fall back to CPU.
async fn calibrate_gpu_validated(config: &distrain_shared::config::NodeConfig) -> Result<u64> {
    let target_interval = config.target_push_interval_secs;
    let min_h = config.min_inner_steps;
    let max_h = config.max_inner_steps;

    let device: GpuDevice = Default::default();
    let model_config = ModelPreset::MicroTest.config();
    let (rope_cos, rope_sin) = precompute_rope_tables::<GpuBackend>(
        model_config.head_dim(),
        model_config.max_seq_len,
        model_config.rope_theta,
        &device,
    );

    let mut module = DistrainTransformerModule::<GpuBackend>::new(&model_config, &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<GpuBackend, DistrainTransformerModule<GpuBackend>>();

    let calibration_steps = 5u64;
    let seq_len = 32usize;
    let batch_size = 2usize;
    let start = Instant::now();

    for step in 0..calibration_steps {
        let tokens: Vec<i64> = (0..(batch_size * seq_len))
            .map(|i| {
                (splitmix64(step * 1000 + i as u64) % model_config.vocab_size as u64) as i64
            })
            .collect();
        let batch = Tensor::<GpuBackend, 2, Int>::from_data(
            TensorData::new(tokens, [batch_size, seq_len]),
            &device,
        );

        let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
        let loss_val: f64 = loss.clone().into_scalar().elem();

        // Validate on first step — a random-init model on vocab=256 should
        // produce loss ≈ ln(256) ≈ 5.5. If the GPU gives 0, NaN, inf, or
        // negative, the compute shaders are broken.
        if step == 0 {
            if loss_val == 0.0 || loss_val.is_nan() || loss_val.is_infinite() || loss_val < 0.0 {
                anyhow::bail!(
                    "GPU produced invalid loss={loss_val} on first calibration step \
                     (expected ~5.5 for random init) — compute shaders are broken"
                );
            }
            info!("GPU validation: first step loss={loss_val:.4} (expected ~5.5) — OK");
        }

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optim.step(3e-4, module, grads_params);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let secs_per_step = elapsed / calibration_steps as f64;
    let h_mini = ((target_interval / secs_per_step) as u64).clamp(min_h, max_h);

    info!("Calibration (GPU): {secs_per_step:.3}s/step → H_mini = {h_mini}");
    Ok(h_mini)
}

/// Train for `inner_steps` on GPU using real data shards.
///
/// Loads checkpoint, trains with AdamW, computes outer delta,
/// compresses it, and writes the compressed bytes to disk.
pub async fn run_training(
    config: &distrain_shared::config::NodeConfig,
    checkpoint_path: &Path,
    inner_steps: u64,
    output_delta_path: &Path,
    data_loader: &mut crate::data::DataLoader,
    error_buffer: &mut ErrorBuffer,
    batch_size: usize,
    grad_accum_steps: usize,
) -> Result<TrainingResult> {
    run_training_with_progress(config, checkpoint_path, inner_steps, output_delta_path, data_loader, error_buffer, batch_size, grad_accum_steps, |_| false).await
}

/// Like `run_training` but calls `on_progress` every 10 steps with intermediate stats.
/// If `on_progress` returns `true`, training aborts early (e.g. new checkpoint available).
pub async fn run_training_with_progress<F>(
    config: &distrain_shared::config::NodeConfig,
    checkpoint_path: &Path,
    inner_steps: u64,
    output_delta_path: &Path,
    data_loader: &mut crate::data::DataLoader,
    error_buffer: &mut ErrorBuffer,
    batch_size: usize,
    grad_accum_steps: usize,
    on_progress: F,
) -> Result<TrainingResult>
where
    F: Fn(StepProgress) -> bool,
{
    if config.force_cpu {
        run_training_with_progress_cpu(config, checkpoint_path, inner_steps, output_delta_path, data_loader, error_buffer, batch_size, grad_accum_steps, on_progress).await
    } else {
        run_training_with_progress_gpu(config, checkpoint_path, inner_steps, output_delta_path, data_loader, error_buffer, batch_size, grad_accum_steps, on_progress).await
    }
}

async fn run_training_with_progress_cpu<F>(
    config: &distrain_shared::config::NodeConfig,
    checkpoint_path: &Path,
    inner_steps: u64,
    output_delta_path: &Path,
    data_loader: &mut crate::data::DataLoader,
    error_buffer: &mut ErrorBuffer,
    batch_size: usize,
    grad_accum_steps: usize,
    on_progress: F,
) -> Result<TrainingResult>
where
    F: Fn(StepProgress) -> bool,
{
    let start_time = Instant::now();
    let device: CpuDevice = Default::default();

    let micro_batch_size = batch_size;
    let effective_batch = micro_batch_size * grad_accum_steps;
    if grad_accum_steps > 1 {
        info!("Gradient accumulation: micro_batch={micro_batch_size}, grad_accum={grad_accum_steps}, effective_batch={effective_batch}");
    }

    crate::resources::log_memory("before checkpoint load");
    let start_params =
        load_safetensors_map(checkpoint_path).context("Failed to load checkpoint")?;
    crate::resources::log_memory("after checkpoint load");
    let model_config = infer_model_config(checkpoint_path)?;

    let (rope_cos, rope_sin) = precompute_rope_tables::<CpuBackend>(
        model_config.head_dim(),
        model_config.max_seq_len,
        model_config.rope_theta,
        &device,
    );

    let module = DistrainTransformerModule::<CpuBackend>::new(&model_config, &device);
    crate::resources::log_memory("after model init");
    let mut module = module.load_state_dict(&start_params, &device);
    crate::resources::log_memory("after weight load");
    let params = config
        .training_params
        .as_ref()
        .cloned()
        .unwrap_or_default();

    let mut optim = AdamWConfig::new()
        .with_weight_decay(params.weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(params.grad_clip_norm as f32)))
        .init::<CpuBackend, DistrainTransformerModule<CpuBackend>>();
    crate::resources::log_memory("after optimizer init");

    let seq_len = config.seq_len;
    let lr_max = params.lr_max;
    let lr_min = params.lr_min;
    let warmup = ((inner_steps as f64 * params.warmup_fraction) as u64).max(2) as usize;

    let mut total_tokens: u64 = 0;
    let mut last_loss: f64 = 0.0;
    let mut loss_sum: f64 = 0.0;
    let mut loss_sq_sum: f64 = 0.0;
    let mut loss_count: u64 = 0;
    let mut steps_done: u64 = 0;
    let mut loss_ema: f64 = 0.0;
    let mut spike_detected = false;

    for step in 0..inner_steps {
        let lr = cosine_lr(step as usize, warmup, inner_steps as usize, lr_max, lr_min);

        for accum in 0..grad_accum_steps {
            let tokens = data_loader.next_batch_sized(micro_batch_size);
            let batch = Tensor::<CpuBackend, 2, Int>::from_data(
                TensorData::new(tokens, [micro_batch_size, seq_len]),
                &device,
            );

            if step == 0 && accum == 0 { crate::resources::log_memory("before first forward"); }
            let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
            let loss_val: f64 = loss.clone().into_scalar().elem();

            if loss_val.is_nan() {
                info!("Step {step} accum {accum}: NaN loss detected, skipping update");
                continue;
            }

            if accum == 0 {
                last_loss = loss_val;
                loss_sum += loss_val;
                loss_sq_sum += loss_val * loss_val;
                loss_count += 1;
                loss_ema = if loss_count == 1 { loss_val } else { loss_ema * 0.9 + loss_val * 0.1 };

                // Spike detection: abort if loss diverges after warmup
                if step as usize > warmup && loss_count > 3 && loss_val > loss_ema * 3.0 && loss_val > loss_ema + 5.0 {
                    warn!("Loss spike: step {} loss={loss_val:.1} vs ema={loss_ema:.1} — aborting round", step + 1);
                    spike_detected = true;
                }
            }

            if step == 0 && accum == 0 { crate::resources::log_memory("after forward, before backward"); }
            let scaled_loss = loss / (grad_accum_steps as f32);
            let grads = scaled_loss.backward();
            if step == 0 && accum == 0 { crate::resources::log_memory("after backward"); }
            let grads_params = GradientsParams::from_grads(grads, &module);
            let lr_step = lr / grad_accum_steps as f64;
            module = optim.step(lr_step, module, grads_params);
            if step == 0 && accum == 0 { crate::resources::log_memory("after optimizer step"); }
        }

        steps_done = step + 1;
        total_tokens += (effective_batch * seq_len) as u64;
        info!("Step {}/{inner_steps}: loss={last_loss:.4}, lr={lr:.2e}", step + 1);
        let should_abort = on_progress(StepProgress {
            step,
            total_steps: inner_steps,
            loss: last_loss,
            lr,
            tokens_processed: total_tokens,
            elapsed_secs: start_time.elapsed().as_secs_f64(),
        });
        if should_abort || spike_detected {
            if spike_detected { info!("Aborting round due to loss spike at step {}/{inner_steps}", step + 1); }
            break;
        }
    }

    let mean_loss = if loss_count > 0 { loss_sum / loss_count as f64 } else { last_loss };
    let loss_variance = if loss_count > 1 {
        loss_sq_sum / loss_count as f64 - mean_loss * mean_loss
    } else { 0.0 };

    let current_params = module.extract_state_dict();
    let shapes = module.extract_shapes();
    let delta = compute_outer_delta(&start_params, &current_params);
    let compression_config = CompressionConfig {
        top_k_fraction: adaptive_top_k(mean_loss),
        ..CompressionConfig::default()
    };
    let (compressed, comp_stats) =
        compress_delta(&delta, &shapes, &compression_config, error_buffer)?;
    let eb_norm: f64 = error_buffer.buffer.values()
        .map(|v| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>())
        .sum::<f64>().sqrt();
    info!("Error buffer norm: {eb_norm:.4} ({} params tracked)", error_buffer.buffer.len());
    info!(
        "Compression: dense_norm={:.4}, sparse_norm={:.4}, retention={:.2}%, kept={}/{}, raw={}MB, compressed={}MB, ratio={:.1}x",
        comp_stats.dense_norm, comp_stats.sparse_norm, comp_stats.retention_ratio * 100.0,
        comp_stats.num_params_kept, comp_stats.num_params_total,
        comp_stats.raw_param_bytes / (1024 * 1024), comp_stats.compressed_bytes / (1024 * 1024),
        comp_stats.raw_param_bytes as f64 / comp_stats.compressed_bytes.max(1) as f64,
    );

    if let Some(parent) = output_delta_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_delta_path, &compressed)
        .context("Failed to write compressed delta")?;

    let elapsed = start_time.elapsed().as_secs_f64();
    info!(
        "Training complete (CPU): {steps_done}/{inner_steps} steps, mean_loss={mean_loss:.4}, last_loss={last_loss:.4}, tokens={total_tokens}, time={elapsed:.1}s"
    );

    Ok(TrainingResult {
        final_loss: mean_loss,
        tokens_processed: total_tokens,
        elapsed_secs: elapsed,
        steps_completed: steps_done,
        batch_size,
        compression_stats: Some(comp_stats),
        loss_spiked: spike_detected,
        loss_variance,
    })
}

/// GPU variant: pre-generates batches from data_loader and delegates to
/// `run_training_gpu_inner` (the single canonical GPU training function).
async fn run_training_with_progress_gpu<F>(
    config: &distrain_shared::config::NodeConfig,
    checkpoint_path: &Path,
    inner_steps: u64,
    output_delta_path: &Path,
    data_loader: &mut crate::data::DataLoader,
    error_buffer: &mut ErrorBuffer,
    batch_size: usize,
    grad_accum_steps: usize,
    on_progress: F,
) -> Result<TrainingResult>
where
    F: Fn(StepProgress) -> bool,
{
    // Pre-generate batches (same pattern as watchdog, single GPU training implementation)
    let batches: Vec<Vec<i64>> = (0..inner_steps * grad_accum_steps as u64)
        .map(|_| data_loader.next_batch_sized(batch_size))
        .collect();

    let eb = std::mem::take(error_buffer);
    let (result, eb_out) = run_training_gpu_inner(
        config, checkpoint_path, inner_steps, output_delta_path,
        batches, eb, batch_size, grad_accum_steps, on_progress,
    ).await?;
    *error_buffer = eb_out;
    Ok(result)
}

/// Calibrate batch size by probing descending sizes until one succeeds.
///
/// GPU: runs gpu-stress-test subprocess for each candidate (crash-safe).
/// CPU: uses memory estimate (no probing needed — CPU won't crash on OOM, just slow).
/// Returns (batch_size, grad_accum_steps, gpu_secs_per_step).
pub async fn calibrate_batch_size(
    checkpoint_path: &Path,
    seq_len: usize,
    use_gpu: bool,
    target_batch: usize,
) -> (usize, usize, Option<f64>) {
    if !use_gpu {
        // CPU: use memory estimate to pick batch size
        let model_config = match infer_model_config(checkpoint_path) {
            Ok(c) => c,
            Err(_) => return (target_batch, 1, None),
        };
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let free_mb = if cfg!(target_os = "macos") {
            sys.total_memory() / (1024 * 1024)
        } else {
            let avail = sys.available_memory();
            (if avail > 0 { avail } else { sys.total_memory().saturating_sub(sys.used_memory()) }) / (1024 * 1024)
        };

        let mut bs = target_batch;
        loop {
            // Estimate: model_params * 20 bytes per batch item for backward pass
            let backward_peak_mb = (model_config.param_count() as u64 * 20 * bs as u64) / (1024 * 1024);
            if backward_peak_mb <= free_mb / 2 || bs <= 1 {
                break;
            }
            bs /= 2;
        }
        let bs = bs.max(1);
        let grad_accum = target_batch / bs;
        if bs < target_batch {
            warn!(
                "CPU batch_size calibration: {target_batch} → {bs} (grad_accum={grad_accum}, {free_mb}MB free)"
            );
        } else {
            info!("CPU batch_size calibration: batch_size={bs} OK ({free_mb}MB free)");
        }
        return (bs, grad_accum.max(1), None);
    }

    // GPU: probe descending batch sizes via subprocess
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return (target_batch, 1, None),
    };

    for &bs in &[8, 4, 2, 1] {
        if bs > target_batch {
            continue;
        }
        info!("Probing GPU batch_size={bs}...");
        let child = tokio::process::Command::new(&exe)
            .arg("gpu-stress-test")
            .arg(checkpoint_path.to_str().unwrap())
            .arg("--batch-size").arg(bs.to_string())
            .arg("--seq-len").arg(seq_len.to_string())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let Ok(child) = child else {
            warn!("Failed to spawn gpu-stress-test for batch_size={bs}");
            continue;
        };

        // 180s timeout: first-ever Metal shader compilation for 1B model takes 30-60s,
        // plus autotuning and the actual forward+backward pass
        match tokio::time::timeout(
            std::time::Duration::from_secs(180),
            child.wait_with_output(),
        ).await {
            Ok(Ok(output)) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().find(|l| l.starts_with("GPU_STRESS_OK")) {
                    if let Some(sps) = line.split_whitespace().nth(1).and_then(|s| s.parse::<f64>().ok()) {
                        let grad_accum = target_batch / bs;
                        info!("GPU batch_size={bs} OK ({sps:.3}s/step), grad_accum={}", grad_accum.max(1));
                        return (bs, grad_accum.max(1), Some(sps));
                    }
                }
            }
            Ok(Ok(output)) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let code = output.status.code().unwrap_or(-1);
                warn!("GPU batch_size={bs} failed (exit {code}): {}", stderr.lines().last().unwrap_or("unknown"));
            }
            Ok(Err(e)) => {
                warn!("GPU batch_size={bs} subprocess error: {e}");
            }
            Err(_) => {
                warn!("GPU batch_size={bs} timed out (120s)");
            }
        }
    }

    // All GPU sizes failed — caller should fall back to CPU
    warn!("All GPU batch sizes failed — recommend CPU fallback");
    (0, 1, None) // batch_size=0 signals failure
}

/// Infer model config from checkpoint tensor shapes.
pub fn infer_model_config(checkpoint_path: &Path) -> Result<ModelConfig> {
    let shapes =
        load_safetensors_shapes(checkpoint_path).context("Failed to load safetensors shapes")?;

    let emb_shape = shapes
        .get("embedding.embedding.weight")
        .context("Missing embedding.embedding.weight in checkpoint")?;
    let vocab_size = emb_shape[0];
    let hidden_dim = emb_shape[1];

    let num_layers = (0..)
        .take_while(|i| shapes.contains_key(&format!("layers.{i}.attn_norm.weight")))
        .count();

    let k_shape = shapes
        .get("layers.0.attention.k_proj.weight")
        .context("Missing layers.0.attention.k_proj.weight")?;
    let kv_dim = k_shape[0];

    let gate_shape = shapes
        .get("layers.0.ffn.gate_proj.weight")
        .context("Missing layers.0.ffn.gate_proj.weight")?;
    let ffn_hidden_dim = gate_shape[0];

    let qkv_bias = shapes.contains_key("layers.0.attention.q_proj.bias");

    for preset in [
        ModelPreset::MicroTest,
        ModelPreset::Tiny,
        ModelPreset::Small,
        ModelPreset::Medium,
        ModelPreset::Large,
    ] {
        let cfg = preset.config();
        if cfg.hidden_dim == hidden_dim
            && cfg.num_layers == num_layers
            && cfg.vocab_size == vocab_size
        {
            info!("Matched checkpoint to preset: {hidden_dim}h/{num_layers}L/{vocab_size}v");
            return Ok(cfg);
        }
    }

    let head_dim = if hidden_dim % 128 == 0 && hidden_dim / 128 >= 2 {
        128
    } else {
        64
    };
    let num_heads = hidden_dim / head_dim;
    let num_kv_heads = kv_dim / head_dim;

    info!(
        "Inferred config: {hidden_dim}h/{num_layers}L/{num_heads}heads/{num_kv_heads}kv/{vocab_size}v"
    );

    Ok(ModelConfig {
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        max_seq_len: 4096,
        ffn_hidden_dim,
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        qkv_bias,
        attention_dropout: 0.0,
        tie_embeddings: true,
    })
}

/// Stress-test the GPU with the **real model** (not MicroTest).
///
/// After downloading the first checkpoint, run 2 forward+backward steps of the
/// actual model on GPU. This catches cases where MicroTest calibration passes
/// but the real model hangs (e.g. Intel Iris Plus on 125M params).
///
/// Returns secs_per_step on success, or `TrainingFailure` on any problem.
pub async fn stress_test_gpu(
    checkpoint_path: &Path,
    config: &distrain_shared::config::NodeConfig,
    timeout_secs: u64,
) -> Result<f64, TrainingFailure> {
    let ckpt_path = checkpoint_path.to_path_buf();
    let batch_size = config.batch_size;
    let seq_len = config.seq_len;

    let gpu_task = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build tokio runtime for GPU stress test");
            rt.block_on(async {
                let device: GpuDevice = Default::default();

                let start_params = load_safetensors_map(&ckpt_path)
                    .context("Failed to load checkpoint for stress test")?;
                let model_config = infer_model_config(&ckpt_path)?;

                let (rope_cos, rope_sin) = precompute_rope_tables::<GpuBackend>(
                    model_config.head_dim(),
                    model_config.max_seq_len,
                    model_config.rope_theta,
                    &device,
                );

                let module = DistrainTransformerModule::<GpuBackend>::new(&model_config, &device);
                let mut module = module.load_state_dict(&start_params, &device);
                let mut optim = AdamWConfig::new()
                    .with_weight_decay(0.1)
                    .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
                    .init::<GpuBackend, DistrainTransformerModule<GpuBackend>>();

                let stress_steps = 1u64;
                let start = Instant::now();

                for step in 0..stress_steps {
                    let tokens: Vec<i64> = (0..(batch_size * seq_len))
                        .map(|i| {
                            (splitmix64(step * 1000 + i as u64) % model_config.vocab_size as u64)
                                as i64
                        })
                        .collect();
                    let batch = Tensor::<GpuBackend, 2, Int>::from_data(
                        TensorData::new(tokens, [batch_size, seq_len]),
                        &device,
                    );

                    let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
                    let loss_val: f64 = loss.clone().into_scalar().elem();

                    if loss_val == 0.0
                        || loss_val.is_nan()
                        || loss_val.is_infinite()
                        || loss_val < 0.0
                    {
                        anyhow::bail!(
                            "GPU produced invalid loss={loss_val} on stress test step {step}"
                        );
                    }
                    if step == 0 {
                        info!(
                            "GPU stress test: step 0 loss={loss_val:.4} — real model OK on GPU"
                        );
                    }

                    let grads = loss.backward();
                    let grads_params = GradientsParams::from_grads(grads, &module);
                    module = optim.step(3e-4, module, grads_params);
                }

                let elapsed = start.elapsed().as_secs_f64();
                let secs_per_step = elapsed / stress_steps as f64;
                info!(
                    "GPU stress test passed: {stress_steps} steps in {elapsed:.1}s ({secs_per_step:.3}s/step)"
                );
                Ok(secs_per_step)
            })
        }))
    });

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        gpu_task,
    )
    .await;

    match result {
        Ok(Ok(Ok(Ok(secs)))) => Ok(secs),
        Ok(Ok(Ok(Err(e)))) => Err(TrainingFailure::Other(e)),
        Ok(Ok(Err(panic_info))) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            Err(TrainingFailure::GpuPanic { message: msg })
        }
        Ok(Err(join_err)) => Err(TrainingFailure::GpuPanic {
            message: format!("task join error: {join_err}"),
        }),
        Err(_) => Err(TrainingFailure::GpuHung {
            timeout_secs: timeout_secs as f64,
        }),
    }
}

/// Measure real-model secs_per_step on CPU.
///
/// Runs 2 forward+backward steps of the actual checkpoint on CPU.
/// No timeout/catch_unwind needed — CPU doesn't hang on driver bugs.
/// Returns secs_per_step.
pub async fn stress_test_cpu(
    checkpoint_path: &Path,
    config: &distrain_shared::config::NodeConfig,
) -> Result<f64> {
    let device: CpuDevice = Default::default();

    let start_params = load_safetensors_map(checkpoint_path)
        .context("Failed to load checkpoint for CPU stress test")?;
    let model_config = infer_model_config(checkpoint_path)?;

    let (rope_cos, rope_sin) = precompute_rope_tables::<CpuBackend>(
        model_config.head_dim(),
        model_config.max_seq_len,
        model_config.rope_theta,
        &device,
    );

    let module = DistrainTransformerModule::<CpuBackend>::new(&model_config, &device);
    let mut module = module.load_state_dict(&start_params, &device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<CpuBackend, DistrainTransformerModule<CpuBackend>>();

    // Use minimal batch for benchmarking to reduce memory spike from gradients.
    // Scale timing estimate to real batch_size/seq_len proportionally.
    let bench_batch = 1usize;
    let bench_seq = 128usize.min(model_config.max_seq_len);
    let scale_factor = (config.batch_size * config.seq_len) as f64 / (bench_batch * bench_seq) as f64;

    let start = Instant::now();
    let tokens: Vec<i64> = (0..(bench_batch * bench_seq))
        .map(|i| {
            (splitmix64(i as u64) % model_config.vocab_size as u64) as i64
        })
        .collect();
    let batch = Tensor::<CpuBackend, 2, Int>::from_data(
        TensorData::new(tokens, [bench_batch, bench_seq]),
        &device,
    );

    let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
    let loss_val: f64 = loss.clone().into_scalar().elem();
    info!("CPU benchmark: loss={loss_val:.4} (batch=1, seq=128)");

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &module);
    let _module = optim.step(3e-4, module, grads_params);

    let bench_secs = start.elapsed().as_secs_f64();
    let secs_per_step = bench_secs * scale_factor;
    info!(
        "CPU benchmark: {bench_secs:.3}s for batch=1/seq=128, estimated {secs_per_step:.3}s/step at batch={}/seq={}",
        config.batch_size, config.seq_len,
    );
    Ok(secs_per_step)
}

/// Run a GPU training round inside a watchdog (timeout + catch_unwind).
///
/// Pre-generates all batches, then runs GPU training inside `spawn_blocking`.
/// Timeout is derived from calibration timing: `h_mini * secs_per_step * 6.0`,
/// minimum 300s. CPU training doesn't need this (CPU doesn't hang on driver bugs).
///
/// The error buffer is moved into the watchdog thread and returned alongside the
/// result, so error feedback accumulates across rounds (critical for top-k compression).
pub async fn run_training_round_with_watchdog<F>(
    config: &distrain_shared::config::NodeConfig,
    checkpoint_path: &Path,
    inner_steps: u64,
    output_delta_path: &Path,
    data_loader: &mut crate::data::DataLoader,
    error_buffer: ErrorBuffer,
    secs_per_step: f64,
    batch_size: usize,
    grad_accum_steps: usize,
    on_progress: F,
) -> (Result<TrainingResult, TrainingFailure>, ErrorBuffer)
where
    F: Fn(StepProgress) -> bool + Send + 'static,
{
    let micro_batch_size = batch_size;
    // Pre-generate all micro-batches on the main thread
    let batches: Vec<Vec<i64>> = (0..inner_steps * grad_accum_steps as u64)
        .map(|_| data_loader.next_batch_sized(micro_batch_size))
        .collect();

    // 6× multiplier: calibration runs in isolated subprocess (fast), but actual training
    // shares unified memory with loaded shards (slower). On M2 Pro with 1B model:
    // calibration = 33s/step, actual = 112s/step (3.3× slower). Use 6× for safety.
    let timeout_secs = (inner_steps as f64 * secs_per_step * 6.0).max(300.0);
    let ckpt_path = checkpoint_path.to_path_buf();
    let delta_path = output_delta_path.to_path_buf();
    let config = config.clone();

    let gpu_task = tokio::task::spawn_blocking(move || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build tokio runtime for GPU training");
            rt.block_on(async {
                run_training_gpu_inner(&config, &ckpt_path, inner_steps, &delta_path, batches, error_buffer, batch_size, grad_accum_steps, on_progress).await
            })
        }))
    });

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs as u64),
        gpu_task,
    )
    .await;

    match result {
        Ok(Ok(Ok(Ok((r, eb))))) => (Ok(r), eb),
        Ok(Ok(Ok(Err(e)))) => (Err(TrainingFailure::Other(e)), ErrorBuffer::new()),
        Ok(Ok(Err(panic_info))) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            (Err(TrainingFailure::GpuPanic { message: msg }), ErrorBuffer::new())
        }
        Ok(Err(join_err)) => (Err(TrainingFailure::GpuPanic {
            message: format!("task join error: {join_err}"),
        }), ErrorBuffer::new()),
        Err(_) => (Err(TrainingFailure::GpuHung { timeout_secs }), ErrorBuffer::new()),
    }
}

/// Inner GPU training loop — called inside spawn_blocking by the watchdog.
async fn run_training_gpu_inner<F>(
    config: &distrain_shared::config::NodeConfig,
    checkpoint_path: &Path,
    inner_steps: u64,
    output_delta_path: &Path,
    batches: Vec<Vec<i64>>,
    mut error_buffer: ErrorBuffer,
    batch_size: usize,
    grad_accum_steps: usize,
    on_progress: F,
) -> Result<(TrainingResult, ErrorBuffer)>
where
    F: Fn(StepProgress) -> bool,
{
    let start_time = Instant::now();
    let device: GpuDevice = Default::default();

    let micro_batch_size = batch_size;
    let effective_batch = micro_batch_size * grad_accum_steps;
    if grad_accum_steps > 1 {
        info!("Gradient accumulation (watchdog): micro_batch={micro_batch_size}, grad_accum={grad_accum_steps}, effective_batch={effective_batch}");
    }

    let start_params =
        load_safetensors_map(checkpoint_path).context("Failed to load checkpoint")?;
    let model_config = infer_model_config(checkpoint_path)?;

    let (rope_cos, rope_sin) = precompute_rope_tables::<GpuBackend>(
        model_config.head_dim(),
        model_config.max_seq_len,
        model_config.rope_theta,
        &device,
    );

    let module = DistrainTransformerModule::<GpuBackend>::new(&model_config, &device);
    let mut module = module.load_state_dict(&start_params, &device);
    let params = config
        .training_params
        .as_ref()
        .cloned()
        .unwrap_or_default();

    let mut optim = AdamWConfig::new()
        .with_weight_decay(params.weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(params.grad_clip_norm as f32)))
        .init::<GpuBackend, DistrainTransformerModule<GpuBackend>>();

    let seq_len = config.seq_len;
    let lr_max = params.lr_max;
    let lr_min = params.lr_min;
    let warmup = ((inner_steps as f64 * params.warmup_fraction) as u64).max(2) as usize;

    let mut total_tokens: u64 = 0;
    let mut last_loss: f64 = 0.0;
    let mut loss_sum: f64 = 0.0;
    let mut loss_sq_sum: f64 = 0.0;
    let mut loss_count: u64 = 0;
    let mut steps_done: u64 = 0;
    let mut loss_ema: f64 = 0.0;
    let mut spike_detected = false;

    // batches is pre-generated as inner_steps * grad_accum_steps micro-batches
    let mut batch_iter = batches.into_iter();

    for step in 0..inner_steps {
        let lr = cosine_lr(step as usize, warmup, inner_steps as usize, lr_max, lr_min);

        for accum in 0..grad_accum_steps {
            let tokens = match batch_iter.next() {
                Some(t) => t,
                None => break,
            };
            let batch = Tensor::<GpuBackend, 2, Int>::from_data(
                TensorData::new(tokens, [micro_batch_size, seq_len]),
                &device,
            );

            let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
            let loss_val: f64 = loss.clone().into_scalar().elem();

            if loss_val.is_nan() {
                info!("Step {step} accum {accum}: NaN loss detected, skipping update");
                continue;
            }

            if accum == 0 {
                last_loss = loss_val;
                loss_sum += loss_val;
                loss_sq_sum += loss_val * loss_val;
                loss_count += 1;
                loss_ema = if loss_count == 1 { loss_val } else { loss_ema * 0.9 + loss_val * 0.1 };

                if step as usize > warmup && loss_count > 3 && loss_val > loss_ema * 3.0 && loss_val > loss_ema + 5.0 {
                    warn!("Loss spike: step {} loss={loss_val:.1} vs ema={loss_ema:.1} — aborting round", step + 1);
                    spike_detected = true;
                }
            }

            let scaled_loss = loss / (grad_accum_steps as f32);
            let grads = scaled_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &module);
            let lr_step = lr / grad_accum_steps as f64;
            module = optim.step(lr_step, module, grads_params);
        }

        steps_done = step + 1;
        total_tokens += (effective_batch * seq_len) as u64;
        info!("Step {}/{inner_steps}: loss={last_loss:.4}, lr={lr:.2e}", step + 1);
        let should_abort = on_progress(StepProgress {
            step,
            total_steps: inner_steps,
            loss: last_loss,
            lr,
            tokens_processed: total_tokens,
            elapsed_secs: start_time.elapsed().as_secs_f64(),
        });
        if should_abort || spike_detected {
            if spike_detected { info!("Aborting round due to loss spike at step {}/{inner_steps}", step + 1); }
            break;
        }
    }

    let mean_loss = if loss_count > 0 { loss_sum / loss_count as f64 } else { last_loss };
    let loss_variance = if loss_count > 1 {
        loss_sq_sum / loss_count as f64 - mean_loss * mean_loss
    } else { 0.0 };

    let current_params = module.extract_state_dict();
    let shapes = module.extract_shapes();
    let delta = compute_outer_delta(&start_params, &current_params);
    let compression_config = CompressionConfig {
        top_k_fraction: adaptive_top_k(mean_loss),
        ..CompressionConfig::default()
    };
    let (compressed, comp_stats) =
        compress_delta(&delta, &shapes, &compression_config, &mut error_buffer)?;
    let eb_norm: f64 = error_buffer.buffer.values()
        .map(|v| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>())
        .sum::<f64>().sqrt();
    info!("Error buffer norm: {eb_norm:.4} ({} params tracked)", error_buffer.buffer.len());
    info!(
        "Compression: dense_norm={:.4}, sparse_norm={:.4}, retention={:.2}%, kept={}/{}, raw={}MB, compressed={}MB, ratio={:.1}x",
        comp_stats.dense_norm, comp_stats.sparse_norm, comp_stats.retention_ratio * 100.0,
        comp_stats.num_params_kept, comp_stats.num_params_total,
        comp_stats.raw_param_bytes / (1024 * 1024), comp_stats.compressed_bytes / (1024 * 1024),
        comp_stats.raw_param_bytes as f64 / comp_stats.compressed_bytes.max(1) as f64,
    );

    if let Some(parent) = output_delta_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_delta_path, &compressed)
        .context("Failed to write compressed delta")?;

    let elapsed = start_time.elapsed().as_secs_f64();
    info!(
        "Training complete (watchdog): {steps_done}/{inner_steps} steps, mean_loss={mean_loss:.4}, last_loss={last_loss:.4}, tokens={total_tokens}, time={elapsed:.1}s"
    );

    Ok((TrainingResult {
        final_loss: mean_loss,
        tokens_processed: total_tokens,
        steps_completed: steps_done,
        elapsed_secs: elapsed,
        batch_size,
        compression_stats: Some(comp_stats),
        loss_spiked: spike_detected,
        loss_variance,
    }, error_buffer))
}

/// SplitMix64 hash — public for subprocess stress test.
pub fn splitmix64_pub(seed: u64) -> u64 {
    splitmix64(seed)
}

/// SplitMix64 hash — only used for calibration timing, not real training.
fn splitmix64(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use distrain_model::{CpuBackend, CpuDevice};

    fn test_config() -> ModelConfig {
        ModelConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            vocab_size: 256,
            max_seq_len: 128,
            ffn_hidden_dim: 128,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
            qkv_bias: true,
            attention_dropout: 0.0,
            tie_embeddings: true,
        }
    }

    #[test]
    fn test_splitmix64_deterministic() {
        assert_eq!(splitmix64(42), splitmix64(42));
        assert_ne!(splitmix64(0), splitmix64(1));
    }

    #[test]
    fn test_native_forward_backward() {
        let device: CpuDevice = Default::default();
        let config = test_config();
        let (rope_cos, rope_sin) = precompute_rope_tables::<CpuBackend>(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            &device,
        );

        let module = DistrainTransformerModule::<CpuBackend>::new(&config, &device);
        let batch = Tensor::<CpuBackend, 2, Int>::from_ints([[1, 2, 3, 4, 5]], &device);

        let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
        let loss_val: f32 = loss.clone().into_scalar().elem();
        assert!(loss_val > 0.0, "Loss should be positive, got {loss_val}");

        let _grads = loss.backward();
    }

    #[test]
    fn test_native_optimizer_step() {
        let device: CpuDevice = Default::default();
        let config = test_config();
        let (rope_cos, rope_sin) = precompute_rope_tables::<CpuBackend>(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            &device,
        );

        let mut module = DistrainTransformerModule::<CpuBackend>::new(&config, &device);
        let mut optim = AdamWConfig::new()
            .init::<CpuBackend, DistrainTransformerModule<CpuBackend>>();

        let batch = Tensor::<CpuBackend, 2, Int>::from_ints([[1, 2, 3, 4, 5]], &device);
        let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optim.step(3e-4, module, grads_params);

        let batch2 = Tensor::<CpuBackend, 2, Int>::from_ints([[5, 4, 3, 2, 1]], &device);
        let loss2 = compute_lm_loss(&module, &rope_cos, &rope_sin, batch2);
        let loss2_val: f32 = loss2.clone().into_scalar().elem();
        assert!(loss2_val > 0.0, "Loss should be positive after step");
    }
}
