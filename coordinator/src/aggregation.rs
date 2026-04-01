//! Pure Rust aggregation — replaces the Python subprocess.
//!
//! Downloads deltas from R2, validates, computes weighted average,
//! applies Nesterov outer optimizer, uploads new checkpoint. All in-memory.

use std::collections::HashMap;

use anyhow::{Context, Result};
use distrain_model::checkpoint::{
    load_safetensors_map_from_bytes, load_safetensors_shapes_from_bytes,
    save_state_dict_safetensors_bytes,
};
use distrain_model::compression::validate_delta;
use distrain_model::training::{weighted_average_deltas, NesterovOuterOptimizer};
use distrain_shared::paths;
use distrain_shared::storage::Storage;
use distrain_shared::types::AccumulatorState;
use tracing::{info, warn};

/// Aggregation metadata written alongside the new checkpoint.
#[derive(serde::Serialize)]
struct AggregationMetadata {
    version: u64,
    contributions_accepted: usize,
    contributions_rejected: usize,
    total_weight: f64,
    outer_delta_norm: f64,
    aggregation_time_secs: f64,
    outer_lr: f64,
}

/// Persisted state for adaptive outer optimizer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OuterLrState {
    outer_lr: f64,
    #[serde(default = "default_momentum")]
    outer_momentum: f64,
    #[serde(default)]
    recent_norms: Vec<f64>,
}

fn default_momentum() -> f64 { 0.9 }

/// Outer LR is fixed at 1.0.
///
/// The merged delta IS the correct update — it's a weighted average of
/// multiple nodes' training results. Scaling it below 1.0 throws away signal.
/// With momentum=0, this means: new_θ = θ_trained (perfect application).
///
/// Previously this was adaptive (0.70-1.0), but every point below 1.0
/// directly reduces gradient signal and widens the gap to single-GPU baseline.
///
/// Returns (1.0, 1.0, loss_ratio) for logging compatibility.
fn compute_loss_based_outer_lr(_current_lr: f64, avg_loss: f64, ln_vocab: f64) -> (f64, f64, f64) {
    let ratio = avg_loss / ln_vocab;
    (1.0, 1.0, ratio)
}

/// Run the full aggregation pipeline in pure Rust.
///
/// 1. Download checkpoint + optimizer velocity + all deltas from R2
/// 2. Decompress and validate each delta
/// 3. Compute weighted average
/// 4. Apply Nesterov outer optimizer step
/// 5. Upload new checkpoint, velocity, and metadata to R2
/// 6. Return (new_version, tokens_this_checkpoint, contributor_node_ids)
pub async fn run_aggregation(
    storage: &Storage,
    acc: &AccumulatorState,
    outer_lr: f64,
    outer_momentum: f64,
    keep_versions: u64,
    vocab_size: u32,
) -> Result<(u64, u64, Vec<String>)> {
    let start = std::time::Instant::now();
    let new_version = acc.checkpoint_version + 1;

    // 1. Download current checkpoint
    let ckpt_key = paths::checkpoint_path(acc.checkpoint_version);
    let ckpt_bytes = storage
        .get(&ckpt_key)
        .await
        .context("Failed to download current checkpoint")?;
    let shapes = load_safetensors_shapes_from_bytes(&ckpt_bytes)
        .context("Failed to parse checkpoint shapes")?;
    let mut checkpoint = load_safetensors_map_from_bytes(&ckpt_bytes)
        .context("Failed to parse checkpoint safetensors")?;
    info!(
        "Loaded checkpoint v{} ({} params)",
        acc.checkpoint_version,
        checkpoint.len()
    );

    // 1b. Compute dynamic outer LR from absolute loss relative to ln(vocab_size)
    let ln_vocab = (vocab_size as f64).ln();
    let mut lr_state = match storage.get_json::<OuterLrState>(&paths::outer_lr_state_path()).await {
        Ok(s) => s,
        Err(_) => OuterLrState { outer_lr, outer_momentum, recent_norms: Vec::new() },
    };

    // Weighted average training loss from contributions
    let total_loss_weight: f64 = acc.contributions.iter().map(|c| c.weight).sum();
    let avg_loss: f64 = if total_loss_weight > 0.0 {
        acc.contributions
            .iter()
            .map(|c| c.training_loss * c.weight)
            .sum::<f64>()
            / total_loss_weight
    } else {
        0.0
    };

    // Compute loss-based outer_lr (skip if no valid loss yet)
    let outer_lr = if avg_loss > 0.0 {
        let (new_lr, target_lr, ratio) =
            compute_loss_based_outer_lr(lr_state.outer_lr, avg_loss, ln_vocab);
        info!(
            "Outer LR: loss={avg_loss:.2}, ratio={ratio:.3}×ln(V), target={target_lr:.3}, applied={new_lr:.4}"
        );
        new_lr
    } else {
        info!("Using initial outer_lr={outer_lr} (no valid loss yet)");
        outer_lr
    };

    // Persist adaptive optimizer state (LR + momentum + norm history)
    lr_state.outer_lr = outer_lr;
    if let Err(e) = storage
        .put_json(&paths::outer_lr_state_path(), &lr_state)
        .await
    {
        warn!("Failed to save outer LR state: {e}");
    }

    // 2. Load optimizer velocity state (may not exist for version 0)
    // outer_opt is created later with adaptive momentum; just load velocity here
    let mut velocity_state: Option<std::collections::HashMap<String, Vec<f32>>> = None;
    let vel_key = paths::optimizer_state_path(acc.checkpoint_version);
    match storage.get(&vel_key).await {
        Ok(vel_bytes) => {
            match load_safetensors_map_from_bytes(&vel_bytes) {
                Ok(vel_state) => {
                    velocity_state = Some(vel_state);
                    info!("Loaded optimizer velocity state");
                }
                Err(e) => warn!("Could not parse optimizer velocity: {e}"),
            }
        }
        Err(_) => info!("No optimizer velocity state found (first aggregation)"),
    }

    // 3. Download all deltas in parallel, then decompress and validate
    let total_weight: f64 = acc.contributions.iter().map(|c| c.weight).sum();
    if total_weight == 0.0 {
        anyhow::bail!("Total weight is zero");
    }

    // Download all deltas concurrently
    let download_futures: Vec<_> = acc
        .contributions
        .iter()
        .map(|contrib| {
            let storage = storage.clone();
            let key = contrib.delta_key.clone();
            async move { (storage.get(&key).await, key) }
        })
        .collect();
    let downloaded: Vec<_> = futures::future::join_all(download_futures).await;

    let mut delta_weight_pairs: Vec<(HashMap<String, Vec<f32>>, f64)> = Vec::new();
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut per_delta_norms: Vec<f64> = Vec::new();
    let mut staleness_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut loss_sum: f64 = 0.0;
    let mut tokens_sum: u64 = 0;

    for (i, contrib) in acc.contributions.iter().enumerate() {
        let delta_bytes = match &downloaded[i] {
            (Ok(b), _) => b,
            (Err(e), key) => {
                warn!("Failed to download delta {key}: {e}");
                rejected += 1;
                continue;
            }
        };

        // Decompress: try zstd first, fallback to raw JSON
        let mut delta = match decompress_delta_bytes(delta_bytes) {
            Ok(d) => d,
            Err(e) => {
                warn!(
                    "Failed to decompress delta from {}: {e}",
                    contrib.node_id
                );
                rejected += 1;
                continue;
            }
        };

        // Validate
        if let Err(reason) = validate_delta(&delta) {
            warn!(
                "Delta from {} failed validation: {reason}",
                contrib.node_id
            );
            rejected += 1;
            continue;
        }

        // Per-delta instrumentation
        let dnorm_sq: f64 = delta.values()
            .map(|v| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>())
            .sum();
        let dnorm = dnorm_sq.sqrt();
        let staleness = acc.checkpoint_version.saturating_sub(contrib.checkpoint_version);
        info!(
            "  delta from {} (v{}): norm={dnorm:.4}, weight={:.4}, steps={}, staleness={}",
            &contrib.node_id.0[..12], contrib.checkpoint_version,
            contrib.weight, contrib.inner_steps, staleness,
        );

        // Delta norm clipping: scale down outliers to 3x running average (preserves direction)
        let mut effective_dnorm = dnorm;
        if !per_delta_norms.is_empty() {
            let avg_norm: f64 = per_delta_norms.iter().sum::<f64>() / per_delta_norms.len() as f64;
            let max_norm = avg_norm * 3.0;
            if dnorm > max_norm && dnorm > 0.0 {
                let scale = max_norm / dnorm;
                info!(
                    "Clipping delta from {} — norm {dnorm:.4} > 3x avg {avg_norm:.4}, scaling by {scale:.4}",
                    &contrib.node_id.0[..12],
                );
                for v in delta.values_mut() {
                    for x in v.iter_mut() {
                        *x *= scale as f32;
                    }
                }
                effective_dnorm = max_norm;
            }
        }

        per_delta_norms.push(effective_dnorm);
        *staleness_counts.entry(staleness).or_insert(0) += 1usize;
        loss_sum += contrib.training_loss;
        tokens_sum += contrib.inner_steps * 4 * 512; // batch_size * seq_len approximation

        delta_weight_pairs.push((delta, contrib.weight));
        accepted += 1;
    }

    info!("Aggregated {accepted} deltas ({rejected} rejected), total_weight={total_weight:.4}");

    if delta_weight_pairs.is_empty() {
        anyhow::bail!("No valid deltas to aggregate");
    }

    // 4. Compute weighted average
    let avg_delta = weighted_average_deltas(&delta_weight_pairs)
        .context("Failed to compute weighted average")?;

    // Compute delta norm for metadata
    let delta_norm_sq: f64 = avg_delta
        .values()
        .map(|v| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>())
        .sum();
    let delta_norm = delta_norm_sq.sqrt();

    // Phase 2+: momentum=0. With high-quality deltas (20% top-k) and few nodes (2-3),
    // momentum amplifies noise more than it helps. Apply the delta directly.
    lr_state.recent_norms.push(delta_norm);
    if lr_state.recent_norms.len() > 10 {
        lr_state.recent_norms.drain(..lr_state.recent_norms.len() - 10);
    }
    let outer_momentum = 0.0;
    lr_state.outer_momentum = outer_momentum;

    info!(
        "Avg delta norm={delta_norm:.4}, outer_lr={outer_lr}, outer_momentum={outer_momentum:.3}"
    );

    // 5. Apply Nesterov outer optimizer step
    let mut outer_opt = NesterovOuterOptimizer::new(outer_lr, outer_momentum);
    if let Some(vel) = velocity_state {
        outer_opt.load_state_dict(vel);
    }
    outer_opt.step(&mut checkpoint, &avg_delta);

    // 6. Upload new checkpoint
    let new_ckpt_bytes = save_state_dict_safetensors_bytes(&checkpoint, &shapes)
        .context("Failed to serialize new checkpoint")?;
    storage
        .put(&paths::checkpoint_path(new_version), new_ckpt_bytes)
        .await
        .context("Failed to upload new checkpoint")?;

    // 7. Upload optimizer velocity state as safetensors
    let vel_state = outer_opt.state_dict();
    if !vel_state.is_empty() {
        let vel_shapes: HashMap<String, Vec<usize>> = vel_state
            .iter()
            .map(|(k, v)| (k.clone(), vec![v.len()]))
            .collect();
        let vel_bytes = save_state_dict_safetensors_bytes(&vel_state, &vel_shapes)
            .context("Failed to serialize velocity state")?;
        storage
            .put(&paths::optimizer_state_path(new_version), vel_bytes)
            .await
            .context("Failed to upload velocity state")?;
    }

    // 8. Upload metadata
    let elapsed = start.elapsed().as_secs_f64();
    let metadata = AggregationMetadata {
        version: new_version,
        contributions_accepted: accepted,
        contributions_rejected: rejected,
        total_weight,
        outer_delta_norm: delta_norm,
        aggregation_time_secs: elapsed,
        outer_lr,
    };
    storage
        .put_json(&paths::checkpoint_metadata_path(new_version), &metadata)
        .await
        .context("Failed to upload metadata")?;

    // Best-effort stats append
    crate::stats::append_stats_entry(
        storage,
        &crate::stats::CheckpointProducedEntry {
            event: "checkpoint_produced",
            timestamp: chrono::Utc::now(),
            version: new_version,
            contributions: accepted,
            total_weight,
            outer_delta_norm: delta_norm,
            aggregation_time_secs: elapsed,
            outer_lr,
            avg_loss: if accepted > 0 { loss_sum / accepted as f64 } else { 0.0 },
            total_tokens: tokens_sum,
            staleness_histogram: {
                let mut h: Vec<(u64, usize)> = staleness_counts.into_iter().collect();
                h.sort_by_key(|(s, _)| *s);
                h
            },
            per_delta_norms,
        },
    )
    .await;

    // Compute tokens and contributor IDs for caller to update in-memory state
    let node_ids: Vec<String> = acc.contributions.iter().map(|c| c.node_id.0.clone()).collect();
    // Each contribution = inner_steps * batch_size(4) * seq_len(512)
    let tokens_this_checkpoint: u64 = acc.contributions.iter().map(|c| c.inner_steps * 4 * 512).sum();

    info!("Aggregation complete: checkpoint v{new_version} in {elapsed:.1}s, delta_norm={delta_norm:.4}");

    // Housekeeping: delete old checkpoints, deltas, and optimizer state from R2.
    // Keep only the last `keep_versions` versions to prevent unbounded storage growth.
    if new_version > keep_versions {
        let delete_up_to = new_version - keep_versions;
        tokio::spawn({
            let storage = storage.clone();
            async move {
                if let Err(e) = cleanup_old_versions(&storage, delete_up_to).await {
                    warn!("Housekeeping failed: {e:#}");
                }
            }
        });
    }

    Ok((new_version, tokens_this_checkpoint, node_ids))
}

/// Delete old checkpoints, deltas, and optimizer state up to (but not including) `up_to_version`.
async fn cleanup_old_versions(storage: &Storage, up_to_version: u64) -> Result<()> {
    let mut deleted = 0u64;
    for v in 0..up_to_version {
        // Try deleting each type — ignore errors (may already be deleted)
        for key in [
            paths::checkpoint_path(v),
            paths::checkpoint_metadata_path(v),
            paths::optimizer_state_path(v),
        ] {
            if storage.delete(&key).await.is_ok() {
                deleted += 1;
            }
        }
        // Delete all deltas for this version (list then delete)
        let prefix = format!("deltas/v{v}/");
        if let Ok(keys) = storage.list_keys(&prefix).await {
            for key in keys {
                let _ = storage.delete(&key).await;
                deleted += 1;
            }
        }
    }
    if deleted > 0 {
        info!("Housekeeping: deleted {deleted} old objects (versions < v{up_to_version})");
    }
    Ok(())
}

/// Decompress delta bytes: try zstd first, then raw JSON.
fn decompress_delta_bytes(data: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    // Try zstd-compressed first (burn-model has zstd-compression as default feature)
    if let Ok(delta) = distrain_model::compression::decompress_delta(data) {
        return Ok(delta);
    }

    // Fallback to raw JSON (for WASM nodes that don't use zstd)
    distrain_model::compression::decompress_delta_json(data)
        .context("Failed to decompress delta (tried zstd and raw JSON)")
}
