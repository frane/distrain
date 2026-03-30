//! R2 state management — read/write accumulator, registry, config.

use anyhow::{Context, Result};
use chrono::Utc;
use distrain_shared::paths;
use distrain_shared::storage::Storage;
use distrain_shared::types::*;
use tracing::info;

/// Load the accumulator state from R2, or create a new empty one.
pub async fn load_accumulator(storage: &Storage) -> Result<AccumulatorState> {
    let key = paths::accumulator_path();
    match storage.get_json::<AccumulatorState>(&key).await {
        Ok(state) => Ok(state),
        Err(_) => {
            info!("No accumulator state found, creating empty");
            Ok(AccumulatorState {
                checkpoint_version: 0,
                contributions: Vec::new(),
                version: 0,
            })
        }
    }
}

/// Save accumulator state to R2.
pub async fn save_accumulator(storage: &Storage, state: &AccumulatorState) -> Result<()> {
    storage
        .put_json(&paths::accumulator_path(), state)
        .await
        .context("Failed to save accumulator state")
}

/// Load node registry from R2.
pub async fn load_registry(storage: &Storage) -> Result<Vec<NodeRegistration>> {
    let key = paths::node_registry_path();
    match storage.get_json::<Vec<NodeRegistration>>(&key).await {
        Ok(reg) => Ok(reg),
        Err(_) => Ok(Vec::new()),
    }
}

/// Save node registry to R2.
pub async fn save_registry(storage: &Storage, registry: &[NodeRegistration]) -> Result<()> {
    storage
        .put_json(&paths::node_registry_path(), &registry)
        .await
        .context("Failed to save node registry")
}

/// Load the latest checkpoint info from R2.
#[allow(dead_code)]
pub async fn load_checkpoint_info(storage: &Storage, version: u64) -> Result<CheckpointInfo> {
    let key = paths::checkpoint_metadata_path(version);
    storage
        .get_json(&key)
        .await
        .context("Failed to load checkpoint info")
}

/// Apply a delta push to the accumulator.
/// Returns (accepted, reason, should_checkpoint).
pub fn apply_delta_push(
    acc: &mut AccumulatorState,
    push: &DeltaPush,
    staleness_decay: f64,
    max_staleness: u64,
) -> (bool, Option<String>, bool) {
    let staleness = acc.checkpoint_version.saturating_sub(push.checkpoint_version);

    // Staleness check
    if staleness > max_staleness {
        return (false, Some(format!("Too stale: {staleness} > {max_staleness}")), false);
    }

    // Idempotent: reject if we have same or newer seq_num from this node
    if let Some(existing) = acc
        .contributions
        .iter()
        .find(|c| c.node_id == push.node_id)
    {
        if existing.seq_num >= push.seq_num {
            return (
                false,
                Some(format!(
                    "Duplicate: existing seq_num {} >= {}",
                    existing.seq_num, push.seq_num
                )),
                false,
            );
        }
        // Remove old contribution from this node (will be replaced)
    }

    // Compute weight
    let staleness_weight = staleness_decay.powi(staleness as i32);
    let step_weight = push.inner_steps as f64 / 50.0;
    let weight = step_weight * staleness_weight;

    // Remove old contribution from this node if present
    acc.contributions.retain(|c| c.node_id != push.node_id);

    // Add new contribution
    acc.contributions.push(ContributionMeta {
        node_id: push.node_id.clone(),
        seq_num: push.seq_num,
        weight,
        checkpoint_version: push.checkpoint_version,
        inner_steps: push.inner_steps,
        delta_key: push.delta_key.clone(),
        received_at: Utc::now(),
        training_loss: push.training_loss,
    });

    acc.version += 1;

    (true, None, false) // should_checkpoint determined by caller
}

/// Patience-based checkpoint trigger.
///
/// Inclusive by design: prefers more contributions (diverse gradients are better),
/// but never stalls. Uses a patience window based on observed round times.
///
/// Logic:
/// 1. Need at least 1 contribution.
/// 2. If MIN_CONTRIBUTIONS is set (> 0) and met → trigger immediately (override).
/// 3. If ALL active nodes have contributed → trigger immediately (best case).
/// 4. Otherwise, wait up to 1.5× median round time for more contributions.
/// 5. If patience expires → trigger with whatever we have (self-healing).
///
/// Dynamic: adapts as nodes join/leave (active = recent heartbeat).
/// Self-healing: patience timeout prevents deadlock from crashed/slow nodes.
pub fn should_checkpoint(
    acc: &AccumulatorState,
    min_contributions_override: u64,
    _min_weight: f64,
    coord_state: &CoordinatorPersistentState,
) -> bool {
    let num_contribs = acc.contributions.len();

    // Need at least 1 contribution
    if num_contribs == 0 {
        return false;
    }

    // MIN_CONTRIBUTIONS override: if set (> 0) and met, trigger immediately.
    // This is a manual override, not the default behavior.
    if min_contributions_override > 0 && num_contribs as u64 >= min_contributions_override {
        return true;
    }

    // Count active nodes (those with recent heartbeats)
    let active_count = coord_state.heartbeats.len();

    // Check how many active nodes have contributed
    let contributing_ids: std::collections::HashSet<&str> = acc.contributions.iter()
        .map(|c| c.node_id.0.as_str())
        .collect();
    let active_contributed = coord_state.heartbeats.keys()
        .filter(|nid| contributing_ids.contains(nid.as_str()))
        .count();

    // All active nodes contributed → trigger immediately
    if active_count > 0 && active_contributed >= active_count {
        return true;
    }

    // Supermajority (>= 2/3 of active) contributed → trigger immediately
    // Better to merge 2 fresh contributions now than wait 2 minutes for the 3rd
    if active_count >= 2 && active_contributed as f64 >= active_count as f64 * 0.66 {
        return true;
    }

    // Patience window: how long since first contribution arrived?
    let now = chrono::Utc::now();
    let first_received = acc.contributions.iter()
        .map(|c| c.received_at)
        .min();
    let elapsed_secs = match first_received {
        Some(first) => (now - first).num_seconds().max(0) as f64,
        None => 0.0,
    };

    // Compute patience from observed round times
    let mut round_times: Vec<f64> = coord_state.node_profiles.values()
        .filter_map(|p| p.round_time_secs)
        .collect();

    let patience_secs = if round_times.is_empty() {
        // Cold start: no timing data. Use generous default (120s).
        120.0
    } else {
        round_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = round_times[round_times.len() / 2];
        // Wait 1.5× median for stragglers, but cap at 5 minutes
        (median * 1.5).min(300.0)
    };

    // Patience expired → trigger with what we have
    if elapsed_secs >= patience_secs {
        return true;
    }

    // Still within patience window, keep waiting for more contributions
    false
}

/// Per-node capability profile tracked by coordinator.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeProfile {
    pub device_type: distrain_shared::types::DeviceType,
    pub vram_mb: u64,
    pub gpu_model: String,
    /// Observed seconds per training round (updated after each delta push).
    pub round_time_secs: Option<f64>,
    /// Tier: "fast", "slow", "very_slow". Computed from round_time relative to median.
    pub tier: String,
}

/// Persistent coordinator state (survives restarts). Stored in R2.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CoordinatorPersistentState {
    #[serde(default)]
    pub total_tokens_trained: u64,
    #[serde(default)]
    pub active_nodes: u64,
    /// (checkpoint_version, vec of node_id strings) for last N checkpoints.
    #[serde(default)]
    pub recent_contributors: Vec<(u64, Vec<String>)>,
    /// node_id -> last heartbeat unix timestamp
    #[serde(default)]
    pub heartbeats: std::collections::HashMap<String, u64>,
    /// node_id -> capability profile (hardware + observed performance)
    #[serde(default)]
    pub node_profiles: std::collections::HashMap<String, NodeProfile>,
}

pub async fn load_coordinator_state(storage: &Storage) -> CoordinatorPersistentState {
    storage
        .get_json(&paths::coordinator_state_path())
        .await
        .unwrap_or_default()
}

