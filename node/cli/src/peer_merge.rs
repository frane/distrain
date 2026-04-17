//! Peer-to-peer delta merging (coordinator-optional mode).
//!
//! When no coordinator is reachable, nodes can merge deltas directly.
//! A node accumulates received peer delta announcements. When enough
//! accumulate (≥ min_contributions), it produces a checkpoint locally
//! using the same weighted average as the coordinator.
//!
//! This makes the coordinator a performance optimization, not a requirement.

use std::collections::HashMap;

use distrain_shared::p2p::types::PeerDeltaAnnouncement;
use tracing::info;

/// Accumulates peer delta announcements for local checkpoint production.
pub struct PeerMergeState {
    /// Received delta metadata (no tensor data — just keys + weights).
    pub received_deltas: Vec<PeerDeltaAnnouncement>,
    /// Minimum contributions before producing a checkpoint.
    pub min_contributions: usize,
    /// Current checkpoint version we're accumulating for.
    pub checkpoint_version: u64,
}

impl PeerMergeState {
    pub fn new(min_contributions: usize, checkpoint_version: u64) -> Self {
        Self {
            received_deltas: Vec::new(),
            min_contributions,
            checkpoint_version,
        }
    }

    /// Add a received peer delta. Returns true if we now have enough to merge.
    pub fn add_delta(&mut self, delta: PeerDeltaAnnouncement) -> bool {
        // Only accept deltas for the current or recent checkpoint version
        if delta.checkpoint_version < self.checkpoint_version.saturating_sub(3) {
            return false;
        }

        // Deduplicate by node_id (keep latest)
        self.received_deltas
            .retain(|d| d.node_id != delta.node_id);
        self.received_deltas.push(delta);

        self.should_produce_checkpoint()
    }

    /// Check if we have enough contributions to produce a checkpoint.
    pub fn should_produce_checkpoint(&self) -> bool {
        self.received_deltas.len() >= self.min_contributions
    }

    /// Take the accumulated deltas and reset for the next round.
    /// Returns (delta_keys, weights) for the coordinator-style merge.
    pub fn take_deltas(&mut self) -> Vec<(String, f64)> {
        let pairs: Vec<(String, f64)> = self
            .received_deltas
            .iter()
            .map(|d| (d.delta_key.clone(), d.weight))
            .collect();

        info!(
            "Peer merge: {} deltas accumulated for v{}, producing checkpoint",
            pairs.len(),
            self.checkpoint_version,
        );

        self.received_deltas.clear();
        pairs
    }

    /// Update the target checkpoint version (after a new checkpoint is produced).
    pub fn advance_version(&mut self, new_version: u64) {
        self.checkpoint_version = new_version;
        // Remove deltas that are too old for the new version
        self.received_deltas.retain(|d| {
            new_version.saturating_sub(d.checkpoint_version) <= 3
        });
    }

    /// Number of accumulated deltas.
    pub fn num_deltas(&self) -> usize {
        self.received_deltas.len()
    }

    /// Unique node IDs that have contributed.
    pub fn contributing_nodes(&self) -> Vec<String> {
        let mut nodes: Vec<String> = self
            .received_deltas
            .iter()
            .map(|d| d.node_id.clone())
            .collect();
        nodes.sort();
        nodes.dedup();
        nodes
    }
}

/// Conflict resolution for concurrent checkpoint production:
/// highest total_contributions wins. Tie-break by version number.
pub fn resolve_checkpoint_conflict(
    local_contributions: u64,
    local_version: u64,
    remote_contributions: u64,
    remote_version: u64,
) -> bool {
    // Returns true if local wins
    if local_contributions != remote_contributions {
        local_contributions > remote_contributions
    } else {
        local_version >= remote_version
    }
}
