//! Domain types shared between coordinator and node client.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Unique node identifier.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub struct NodeId(pub String);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Node registration record stored in R2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRegistration {
    pub node_id: NodeId,
    pub gpu_model: String,
    pub gpu_memory_gb: f64,
    pub bandwidth_mbps: f64,
    pub registered_at: DateTime<Utc>,
    pub status: NodeStatus,
    pub api_key_hash: String,
    pub last_seen: DateTime<Utc>,
    pub total_contributions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Pending,
    Active,
    Suspended,
    Offline,
}

/// Delta push request from node to coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaPush {
    pub node_id: NodeId,
    pub seq_num: u64,
    pub checkpoint_version: u64,
    pub inner_steps: u64,
    pub delta_key: String,
    pub training_loss: f64,
    pub tokens_processed: u64,
    pub training_time_secs: f64,
    /// Compressed delta size in bytes (for paper metrics).
    #[serde(default)]
    pub compressed_bytes: Option<u64>,
    /// L2 norm of dense delta before top-k sparsification.
    #[serde(default)]
    pub dense_norm: Option<f64>,
    /// L2 norm of sparse delta after top-k sparsification.
    #[serde(default)]
    pub sparse_norm: Option<f64>,
}

/// Response to delta push.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaPushResponse {
    pub accepted: bool,
    pub checkpoint_version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Info about the latest checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub version: u64,
    pub checkpoint_key: String,
    pub metadata_key: String,
    pub val_loss: Option<f64>,
    pub total_contributions: u64,
    pub total_tokens: u64,
    pub created_at: DateTime<Utc>,
}

/// CRDT accumulator state persisted to R2 as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccumulatorState {
    pub checkpoint_version: u64,
    pub contributions: Vec<ContributionMeta>,
    pub version: u64,
}

/// Metadata about a single delta contribution (no tensors, just keys).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionMeta {
    pub node_id: NodeId,
    pub seq_num: u64,
    pub weight: f64,
    pub checkpoint_version: u64,
    pub inner_steps: u64,
    pub delta_key: String,
    pub received_at: DateTime<Utc>,
    #[serde(default)]
    pub training_loss: f64,
}

/// Global training configuration stored in R2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub model_size: String,
    pub min_contributions: u64,
    pub staleness_decay: f64,
    pub max_staleness: u64,
    pub outer_lr: f64,
    pub outer_momentum: f64,
    pub default_inner_steps: u64,
    pub current_checkpoint_version: u64,
    pub total_tokens_trained: u64,
    pub total_shards: u32,
    pub shards_per_node: u32,
}

/// Public training status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub checkpoint_version: u64,
    pub active_nodes: u64,
    pub total_contributions: u64,
    pub total_tokens_trained: u64,
    pub accumulator_contributions: u64,
    pub latest_val_loss: Option<f64>,
    pub loss_history: Vec<(u64, f64)>,
    /// node_id -> last heartbeat unix timestamp
    #[serde(default)]
    pub node_last_seen: Vec<(String, u64)>,
}

/// Canonical training hyperparameters distributed by the coordinator.
///
/// All node platforms (CLI, Desktop, Browser) use these defaults unless
/// overridden locally (e.g. browser uses smaller batch_size/seq_len for WASM memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    pub batch_size: usize,
    pub seq_len: usize,
    pub lr_max: f64,
    pub lr_min: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub warmup_fraction: f64,
    pub shards_fraction: f64,
    #[serde(default = "default_min_inner_steps")]
    pub min_inner_steps: u64,
    #[serde(default = "default_max_inner_steps")]
    pub max_inner_steps: u64,
    #[serde(default = "default_target_push_interval")]
    pub target_push_interval_secs: f64,
}

fn default_min_inner_steps() -> u64 { 50 }
fn default_max_inner_steps() -> u64 { 500 }
fn default_target_push_interval() -> f64 { 60.0 }

impl TrainingParams {
    /// Compute the number of shards this node should train on per round.
    pub fn shards_per_node(&self, total_shards: usize) -> usize {
        ((total_shards as f64 * self.shards_fraction) as usize)
            .max(2)
            .min(total_shards)
    }
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            batch_size: 4,
            seq_len: 512,
            lr_max: 3e-4,
            lr_min: 1e-6,
            weight_decay: 0.1,
            grad_clip_norm: 1.0,
            warmup_fraction: 0.2,
            shards_fraction: 0.2,
            min_inner_steps: 50,
            max_inner_steps: 500,
            target_push_interval_secs: 60.0,
        }
    }
}

/// Heartbeat request from node to coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub node_id: NodeId,
    #[serde(default)]
    pub step: Option<u64>,
    #[serde(default)]
    pub total_steps: Option<u64>,
    #[serde(default)]
    pub loss: Option<f64>,
    #[serde(default)]
    pub checkpoint_version: Option<u64>,
}

/// Heartbeat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub active_nodes: u64,
    /// If true, node should abort current round and pull new checkpoint.
    #[serde(default)]
    pub should_abort: bool,
    #[serde(default)]
    pub latest_version: Option<u64>,
}

/// Type of compute device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    Cpu,
    Unknown,
}

impl Default for DeviceType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Hardware profile reported by a node at registration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareProfile {
    pub gpu_model: String,
    pub vram_mb: u64,
    pub device_type: DeviceType,
    pub cpu_cores: u32,
    pub ram_mb: u64,
    #[serde(default)]
    pub measured_step_time_secs: Option<f64>,
}

/// Node registration request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub gpu_model: String,
    pub gpu_memory_gb: f64,
    pub bandwidth_mbps: f64,
    /// Persistent node ID — reused across restarts to avoid ghost node inflation.
    #[serde(default)]
    pub node_id: Option<String>,
    /// Detailed hardware profile for coordinator-side analytics.
    #[serde(default)]
    pub hardware: Option<HardwareProfile>,
}

/// Node registration response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub node_id: NodeId,
    pub api_key: String,
    pub status: NodeStatus,
    /// S3-compatible storage endpoint for delta uploads and checkpoint downloads.
    #[serde(default)]
    pub storage_endpoint: Option<String>,
    /// S3 bucket name.
    #[serde(default)]
    pub storage_bucket: Option<String>,
    /// Canonical training hyperparameters from the coordinator.
    #[serde(default)]
    pub training_params: Option<TrainingParams>,
}

/// Auto-discovery response: everything a node needs to join the training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAutoConfig {
    pub storage: StorageConfigPublic,
    pub training_params: TrainingParams,
    pub coordinator_version: String,
}

/// Public-facing storage configuration (returned by coordinator for auto-discovery).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfigPublic {
    pub endpoint: String,
    pub bucket: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: String,
}
