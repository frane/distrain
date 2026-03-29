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

/// Capability-based checkpoint trigger.
///
/// Classifies active nodes into tiers based on observed round times:
/// - Fast: round_time < 2× median (these nodes block checkpoint production)
/// - Slow: 2-10× median (contribute but don't block)
/// - Very slow: >10× median (best-effort, never block)
///
/// Triggers checkpoint when all fast-tier nodes have contributed.
/// Falls back to min_contributions if no profiles exist yet (cold start).
pub fn should_checkpoint(
    acc: &AccumulatorState,
    min_contributions: u64,
    _min_weight: f64,
    coord_state: &CoordinatorPersistentState,
) -> bool {
    // Hard floor: always need at least min_contributions
    if (acc.contributions.len() as u64) < min_contributions {
        return false;
    }

    // Get active nodes (those with recent heartbeats) that have profiles
    let active_node_ids: std::collections::HashSet<&String> = coord_state.heartbeats.keys().collect();
    let active_profiles: Vec<(&String, &NodeProfile)> = coord_state.node_profiles.iter()
        .filter(|(nid, _)| active_node_ids.contains(nid))
        .collect();

    // Cold start: no profiles yet, fall back to count-based
    if active_profiles.is_empty() {
        return true; // min_contributions already passed
    }

    // Compute median round time from nodes that have observed times
    let mut round_times: Vec<f64> = active_profiles.iter()
        .filter_map(|(_, p)| p.round_time_secs)
        .collect();

    if round_times.is_empty() {
        return true; // no timing data yet, fall back to count-based
    }

    round_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = round_times[round_times.len() / 2];

    // Classify: fast nodes are those with round_time < 2× median
    let fast_node_ids: std::collections::HashSet<&String> = active_profiles.iter()
        .filter(|(_, p)| {
            match p.round_time_secs {
                Some(t) => t < median * 2.0,
                None => true, // no timing yet = assume fast (don't skip new nodes)
            }
        })
        .map(|(nid, _)| *nid)
        .collect();

    // Check: have all fast nodes contributed?
    let contributing_node_ids: std::collections::HashSet<&str> = acc.contributions.iter()
        .map(|c| c.node_id.0.as_str())
        .collect();

    let fast_contributed = fast_node_ids.iter()
        .all(|nid| contributing_node_ids.contains(nid.as_str()));

    fast_contributed
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

