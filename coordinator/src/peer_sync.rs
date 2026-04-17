//! Coordinator-to-coordinator merge via gossipsub.
//!
//! When another coordinator announces a checkpoint, this coordinator
//! downloads it and merges via weighted averaging:
//!
//!   combined = (A.checkpoint × A.contributions + B.checkpoint × B.contributions)
//!              / (A.contributions + B.contributions)
//!
//! The result is published as a new version. Conflict resolution:
//! highest total_contributions wins.

use std::collections::HashMap;

use anyhow::{Context, Result};
use distrain_model::checkpoint::{
    load_safetensors_map_from_bytes, load_safetensors_shapes_from_bytes,
    save_state_dict_safetensors_bytes,
};
use distrain_shared::p2p::types::CoordinatorSyncMessage;
use distrain_shared::paths;
use distrain_shared::storage::Storage;
use tracing::info;

/// Merge a remote coordinator's checkpoint with the local checkpoint.
///
/// Downloads the remote checkpoint, computes a weighted average with
/// the local checkpoint, and uploads the merged result as a new version.
pub async fn merge_remote_checkpoint(
    storage: &Storage,
    local_version: u64,
    remote_msg: &CoordinatorSyncMessage,
) -> Result<u64> {
    let new_version = local_version.max(remote_msg.checkpoint_version) + 1;

    // Download local checkpoint
    let local_bytes = storage
        .get(&paths::checkpoint_path(local_version))
        .await
        .context("Failed to download local checkpoint")?;
    let local_shapes = load_safetensors_shapes_from_bytes(&local_bytes)?;
    let local_params = load_safetensors_map_from_bytes(&local_bytes)?;

    // Download remote checkpoint
    let remote_bytes = storage
        .get(&remote_msg.checkpoint_r2_path)
        .await
        .context("Failed to download remote checkpoint")?;
    let remote_params = load_safetensors_map_from_bytes(&remote_bytes)?;

    // Weighted average: weight by number of contributions
    let local_weight = 1.0f64; // local contributions already applied
    let remote_weight = remote_msg.num_contributions as f64;
    let total_weight = local_weight + remote_weight;

    if total_weight == 0.0 {
        anyhow::bail!("Total weight is zero");
    }

    let local_frac = (local_weight / total_weight) as f32;
    let remote_frac = (remote_weight / total_weight) as f32;

    let mut merged: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, local_vals) in &local_params {
        if let Some(remote_vals) = remote_params.get(name) {
            if local_vals.len() == remote_vals.len() {
                let m: Vec<f32> = local_vals
                    .iter()
                    .zip(remote_vals.iter())
                    .map(|(l, r)| l * local_frac + r * remote_frac)
                    .collect();
                merged.insert(name.clone(), m);
            } else {
                // Shape mismatch — keep local
                merged.insert(name.clone(), local_vals.clone());
            }
        } else {
            merged.insert(name.clone(), local_vals.clone());
        }
    }

    // Upload merged checkpoint
    let merged_bytes = save_state_dict_safetensors_bytes(&merged, &local_shapes)?;
    storage
        .put(&paths::checkpoint_path(new_version), merged_bytes)
        .await
        .context("Failed to upload merged checkpoint")?;

    info!(
        "Merged remote checkpoint (v{} from {}, {} contributions) with local v{} → v{}",
        remote_msg.checkpoint_version,
        remote_msg.coordinator_id,
        remote_msg.num_contributions,
        local_version,
        new_version,
    );

    Ok(new_version)
}
