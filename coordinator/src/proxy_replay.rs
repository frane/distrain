//! Proxy replay bulletin board.
//!
//! When a very stale delta arrives (staleness 13+), the coordinator rejects
//! it from the merge but saves the data manifest (which shards, offsets the
//! slow node trained on) to a bulletin board in R2.
//!
//! Fast nodes that finish a round and have idle capacity voluntarily check
//! the board, grab a request, train on that data against the current
//! checkpoint, and push a fresh delta. No assignment, no obligation.

use chrono::{DateTime, Duration, Utc};
use distrain_shared::storage::Storage;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// A replay request posted to the bulletin board.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyReplayRequest {
    /// Node that produced the too-stale delta.
    pub source_node_id: String,
    /// Shard IDs the node trained on.
    pub shard_ids: Vec<u32>,
    /// Token offset within each shard.
    pub shard_offsets: Vec<u64>,
    /// The checkpoint version to train against (current at time of rejection).
    pub target_checkpoint_version: u64,
    /// When the request was posted.
    pub timestamp: DateTime<Utc>,
    /// When the request expires (auto-cleanup).
    pub expires_at: DateTime<Utc>,
    /// Staleness of the rejected delta.
    pub staleness: u64,
    /// Training loss reported by the slow node (for prioritization).
    pub training_loss: f64,
    /// Number of inner steps the slow node completed.
    pub inner_steps: u64,
}

/// Write a replay request to the bulletin board in R2.
pub async fn post_replay_request(
    storage: &Storage,
    request: &ProxyReplayRequest,
) {
    let key = format!(
        "replay_board/{}_{}.json",
        request.timestamp.format("%Y%m%dT%H%M%S"),
        &request.source_node_id,
    );

    match storage.put_json(&key, request).await {
        Ok(()) => {
            info!(
                "Posted replay request: node={}, {} shards, target v{}, expires {}",
                &request.source_node_id,
                request.shard_ids.len(),
                request.target_checkpoint_version,
                request.expires_at.format("%H:%M:%S"),
            );
        }
        Err(e) => {
            warn!("Failed to post replay request: {e}");
        }
    }
}

/// List all pending replay requests from the board.
/// Returns requests sorted by training_loss (highest loss = most value to replay).
/// Filters out expired requests.
pub async fn list_replay_requests(storage: &Storage) -> Vec<ProxyReplayRequest> {
    let keys = match storage.list_keys("replay_board/").await {
        Ok(k) => k,
        Err(e) => {
            warn!("Failed to list replay board: {e}");
            return Vec::new();
        }
    };

    let now = Utc::now();
    let mut requests = Vec::new();

    for key in &keys {
        if let Ok(req) = storage.get_json::<ProxyReplayRequest>(key).await {
            if req.expires_at > now {
                requests.push(req);
            }
        }
    }

    // Highest loss first (most value to replay)
    requests.sort_by(|a, b| b.training_loss.partial_cmp(&a.training_loss).unwrap_or(std::cmp::Ordering::Equal));
    requests
}

/// Claim a replay request by deleting it from the board.
/// Returns the request if it still existed (first-come-first-served).
pub async fn claim_replay_request(
    storage: &Storage,
    request: &ProxyReplayRequest,
) -> bool {
    let key = format!(
        "replay_board/{}_{}.json",
        request.timestamp.format("%Y%m%dT%H%M%S"),
        &request.source_node_id,
    );

    match storage.delete(&key).await {
        Ok(()) => {
            info!(
                "Claimed replay request: node={}, target v{}",
                request.source_node_id, request.target_checkpoint_version,
            );
            true
        }
        Err(_) => false, // already claimed by another node
    }
}

/// Clean up expired replay requests from the board.
pub async fn cleanup_expired(storage: &Storage) {
    let keys = match storage.list_keys("replay_board/").await {
        Ok(k) => k,
        Err(_) => return,
    };

    let now = Utc::now();
    let mut cleaned = 0;

    for key in &keys {
        if let Ok(req) = storage.get_json::<ProxyReplayRequest>(key).await {
            if req.expires_at <= now {
                let _ = storage.delete(key).await;
                cleaned += 1;
            }
        }
    }

    if cleaned > 0 {
        info!("Replay board cleanup: removed {cleaned} expired requests");
    }
}

/// Create a replay request from a rejected stale delta.
pub fn create_replay_request(
    source_node_id: &str,
    target_checkpoint_version: u64,
    staleness: u64,
    training_loss: f64,
    inner_steps: u64,
    expiry_hours: i64,
) -> ProxyReplayRequest {
    let now = Utc::now();
    ProxyReplayRequest {
        source_node_id: source_node_id.to_string(),
        shard_ids: Vec::new(),     // filled by caller if shard info available
        shard_offsets: Vec::new(),  // filled by caller if offset info available
        target_checkpoint_version,
        timestamp: now,
        expires_at: now + Duration::hours(expiry_hours),
        staleness,
        training_loss,
        inner_steps,
    }
}
