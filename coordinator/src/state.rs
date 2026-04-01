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
                contributions: Vec::new(), first_contribution_at: None,
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

    // Weight = tokens processed × staleness decay.
    // More tokens = more signal. Stale deltas get exponentially less weight.
    let staleness_weight = staleness_decay.powi(staleness as i32);
    let weight = push.tokens_processed as f64 * staleness_weight;

    // Remove old contribution from this node if present
    acc.contributions.retain(|c| c.node_id != push.node_id);

    // Add new contribution
    acc.contributions.push(ContributionMeta {
        node_id: push.node_id.clone(),
        seq_num: push.seq_num,
        weight,
        checkpoint_version: push.checkpoint_version,
        inner_steps: push.inner_steps,
        tokens_processed: push.tokens_processed,
        delta_key: push.delta_key.clone(),
        received_at: Utc::now(),
        training_loss: push.training_loss,
    });

    // Set patience start time on first contribution (never reset within cycle)
    if acc.first_contribution_at.is_none() {
        acc.first_contribution_at = Some(Utc::now());
    }

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

    // Patience window: how long since first contribution in this cycle?
    // Uses first_contribution_at which is set once and never reset by subsequent pushes.
    let now = chrono::Utc::now();
    let elapsed_secs = match acc.first_contribution_at {
        Some(first) => (now - first).num_seconds().max(0) as f64,
        None => 0.0,
    };

    // Compute patience from ACTIVE nodes' expected round times.
    // Prefer node-reported expected_round_time (from calibration) over observed push intervals.
    let active_ids: std::collections::HashSet<&String> = coord_state.heartbeats.keys().collect();
    let mut round_times: Vec<f64> = coord_state.node_profiles.iter()
        .filter(|(nid, _)| active_ids.contains(nid))
        .filter_map(|(_, p)| {
            // Use reported expected time if available, fall back to observed
            p.expected_round_time.or(p.round_time_secs)
        })
        .collect();

    let patience_secs = if round_times.is_empty() {
        // Cold start: no timing data. Use moderate default (60s).
        // Short enough to not stall, long enough for 3 nodes to push.
        60.0
    } else {
        round_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = round_times[round_times.len() / 2];
        // Wait for nodes within 3× median (the "cluster" of similar-speed nodes).
        // Outliers beyond 3× are too slow to wait for — they contribute when ready.
        // This way: 3 identical GPUs → wait for all. 2 GPUs + 1 CPU → wait for GPUs only.
        let cluster_max = round_times.iter()
            .filter(|&&t| t <= median * 3.0)
            .copied()
            .last()
            .unwrap_or(median);
        (cluster_max * 1.2).clamp(30.0, 300.0)
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
    /// Observed seconds per training round (time between consecutive delta pushes).
    pub round_time_secs: Option<f64>,
    /// Node-reported expected round time (from calibration: h_mini * step_time + overhead).
    #[serde(default)]
    pub expected_round_time: Option<f64>,
    /// Node-reported step time in seconds.
    #[serde(default)]
    pub step_time_secs: Option<f64>,
    /// Node-reported H_mini.
    #[serde(default)]
    pub h_mini: Option<u64>,
    /// Unix timestamp of last delta push from this node.
    #[serde(default)]
    pub last_push_time: Option<u64>,
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

