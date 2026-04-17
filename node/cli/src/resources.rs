//! Memory budget estimation — clamps shard count to fit available RAM.

use std::fmt;

use anyhow::{bail, Result};
use sysinfo::System;
use tracing::info;

use distrain_model::config::ModelConfig;
use distrain_shared::config::NodeConfig;

/// Estimated memory budget for a training run.
pub struct MemoryBudget {
    pub available_mb: u64,
    pub usable_mb: u64,
    pub model_mb: u64,
    pub optimizer_mb: u64,
    pub per_shard_mb: u64,
    pub max_shards: usize,
}

impl fmt::Display for MemoryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryBudget {{ available={}MB, usable={}MB, model={}MB, optimizer={}MB, per_shard={}MB, max_shards={} }}",
            self.available_mb, self.usable_mb, self.model_mb, self.optimizer_mb, self.per_shard_mb, self.max_shards,
        )
    }
}

/// Estimate available memory and compute how many shards fit.
///
/// Uses `sysinfo` to query available physical memory. Reserves a headroom
/// fraction (configurable via `NodeConfig::max_memory_fraction`, default 0.80).
///
/// Memory model:
/// - Model weights: `param_count * 4` bytes (FP32 during training)
/// - Optimizer (AdamW): `param_count * 8` bytes (momentum + variance, FP32 each)
/// - Activation buffer: ~25% of model size (rough estimate)
/// - Per shard: ~20MB (10M tokens × 2 bytes/token)
pub fn compute_memory_budget(
    model_config: &ModelConfig,
    config: &NodeConfig,
) -> Result<MemoryBudget> {
    let mut sys = System::new();
    sys.refresh_memory();

    // macOS unified memory: sysinfo's available_memory() excludes cached/inactive
    // pages, hugely underestimating reclaimable memory (e.g. 10GB "available" on 32GB).
    // On macOS, use total memory as budget base — the OS reclaims cached pages on demand.
    // On Linux, available_memory() is accurate and should be used.
    let available_bytes = if cfg!(target_os = "macos") {
        sys.total_memory()
    } else {
        let avail = sys.available_memory();
        if avail > 0 { avail } else { sys.total_memory().saturating_sub(sys.used_memory()) }
    };
    let available_mb = available_bytes / (1024 * 1024);
    let fraction = config.max_memory_fraction.clamp(0.1, 1.0);
    let usable_mb = (available_mb as f64 * fraction) as u64;

    let param_count = model_config.param_count() as u64;
    // FP32 weights during training
    let model_mb = param_count * 4 / (1024 * 1024);
    // AdamW: momentum + variance (each FP32 = 4 bytes per param)
    let optimizer_mb = param_count * 8 / (1024 * 1024);
    // Rough activation buffer estimate (~25% of model)
    let activation_mb = model_mb / 4;

    let overhead_mb = model_mb + optimizer_mb + activation_mb;

    if overhead_mb > usable_mb {
        bail!(
            "Model + optimizer requires ~{}MB but only {}MB usable ({}MB available × {:.0}% fraction). \
             Reduce model size or free memory.",
            overhead_mb, usable_mb, available_mb, fraction * 100.0,
        );
    }

    let remaining_mb = usable_mb - overhead_mb;
    let per_shard_mb: u64 = 20; // 10M tokens × 2 bytes ≈ 20MB
    let max_shards = (remaining_mb / per_shard_mb) as usize;

    info!(
        "Memory: {}MB available, {}MB usable ({:.0}%), model={}MB, optimizer={}MB, activations={}MB → room for {} shards",
        available_mb, usable_mb, fraction * 100.0, model_mb, optimizer_mb, activation_mb, max_shards,
    );

    Ok(MemoryBudget {
        available_mb,
        usable_mb,
        model_mb,
        optimizer_mb,
        per_shard_mb,
        max_shards,
    })
}

/// Log current memory usage with a label. Cheap to call.
pub fn log_memory(label: &str) {
    let mut sys = System::new();
    sys.refresh_memory();
    let total = sys.total_memory() / (1024 * 1024);
    let used = sys.used_memory() / (1024 * 1024);
    let free = total.saturating_sub(used);
    info!("[mem] {label}: {used}MB used / {total}MB total ({free}MB free)");
}

/// VRAM-based training strategy for heterogeneous devices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VramStrategy {
    /// >= 24 GB: full model in VRAM, large batch.
    FullModel,
    /// >= 16 GB: gradient checkpointing enabled, medium batch.
    GradientCheckpointing,
    /// >= 8 GB: gradient checkpointing + optimizer state offload to CPU RAM.
    CheckpointWithOffload,
    /// < 8 GB: CPU-only training.
    CpuOnly,
}

impl std::fmt::Display for VramStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VramStrategy::FullModel => write!(f, "full_model (VRAM >= 24GB)"),
            VramStrategy::GradientCheckpointing => write!(f, "gradient_checkpointing (VRAM >= 16GB)"),
            VramStrategy::CheckpointWithOffload => write!(f, "checkpoint_with_offload (VRAM >= 8GB)"),
            VramStrategy::CpuOnly => write!(f, "cpu_only (VRAM < 8GB)"),
        }
    }
}

/// Determine optimal training strategy based on available VRAM.
///
/// Returns the strategy and suggested batch size scaling factor.
pub fn determine_vram_strategy(vram_mb: u64, model_config: &ModelConfig) -> (VramStrategy, f64) {
    let param_count = model_config.param_count() as u64;
    // BF16 model weights = 2 bytes per param
    let model_vram_mb = param_count * 2 / (1024 * 1024);
    // Training overhead (gradients, optimizer, activations) ~4x model weights
    let training_vram_mb = model_vram_mb * 4;

    info!(
        "VRAM cascade: {vram_mb}MB available, model={model_vram_mb}MB, training_estimate={training_vram_mb}MB"
    );

    if vram_mb >= 24 * 1024 {
        // Lots of VRAM: full model, large batch
        info!("VRAM strategy: full_model — large batch enabled");
        (VramStrategy::FullModel, 1.0)
    } else if vram_mb >= 16 * 1024 {
        // Medium VRAM: gradient checkpointing reduces activation memory ~60%
        info!("VRAM strategy: gradient_checkpointing — activation memory reduced");
        (VramStrategy::GradientCheckpointing, 0.75)
    } else if vram_mb >= 8 * 1024 {
        // Low VRAM: checkpoint + offload optimizer to CPU RAM
        info!("VRAM strategy: checkpoint_with_offload — optimizer offloaded to CPU");
        (VramStrategy::CheckpointWithOffload, 0.5)
    } else {
        // Very low VRAM: CPU-only training
        info!("VRAM strategy: cpu_only — VRAM {vram_mb}MB insufficient for GPU training");
        (VramStrategy::CpuOnly, 0.25)
    }
}

/// Check if system memory usage exceeds `max_fraction + 10%` (abort threshold).
///
/// Returns `true` if memory pressure is critical and the caller should shed load.
/// Cheap to call (~microseconds) — suitable for per-step polling.
pub fn check_memory_pressure(max_fraction: f64) -> bool {
    let mut sys = System::new();
    sys.refresh_memory();

    let total = sys.total_memory();
    if total == 0 {
        return false;
    }

    // macOS: available_memory() returns 0 in sysinfo 0.33. Use used_memory() directly.
    let used = sys.used_memory();
    let fraction = used as f64 / total as f64;
    let threshold = (max_fraction + 0.10).min(1.0);

    fraction > threshold
}
