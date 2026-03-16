//! Append-only JSONL stats metastore in R2.
//!
//! Persists delta/checkpoint metadata that housekeeping deletes.
//! Best-effort — failures are logged but never block the main flow.

use chrono::{DateTime, Utc};
use distrain_shared::paths;
use distrain_shared::storage::Storage;
use distrain_shared::types::NodeId;
use serde::Serialize;
use tracing::warn;

#[derive(Serialize)]
pub struct DeltaAcceptedEntry {
    pub event: &'static str,
    pub timestamp: DateTime<Utc>,
    pub node_id: NodeId,
    pub seq_num: u64,
    pub checkpoint_version: u64,
    pub inner_steps: u64,
    pub training_loss: f64,
    pub weight: f64,
    pub staleness: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_processed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compressed_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse_norm: Option<f64>,
}

#[derive(Serialize)]
pub struct CheckpointProducedEntry {
    pub event: &'static str,
    pub timestamp: DateTime<Utc>,
    pub version: u64,
    pub contributions: usize,
    pub total_weight: f64,
    pub outer_delta_norm: f64,
    pub aggregation_time_secs: f64,
    pub outer_lr: f64,
    pub avg_loss: f64,
    pub total_tokens: u64,
    /// (staleness, count) pairs
    pub staleness_histogram: Vec<(u64, usize)>,
    pub per_delta_norms: Vec<f64>,
}

pub async fn append_stats_entry<T: Serialize>(storage: &Storage, entry: &T) {
    let line = match serde_json::to_string(entry) {
        Ok(l) => l,
        Err(e) => {
            warn!("Failed to serialize stats entry: {e}");
            return;
        }
    };

    let key = paths::stats_history_path();

    // Download existing, append, re-upload (best-effort)
    let mut existing = match storage.get(&key).await {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => String::new(),
    };

    if !existing.is_empty() && !existing.ends_with('\n') {
        existing.push('\n');
    }
    existing.push_str(&line);
    existing.push('\n');

    if let Err(e) = storage.put(&key, existing.into_bytes()).await {
        warn!("Failed to append stats entry: {e}");
    }
}
