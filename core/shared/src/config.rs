//! Configuration types for R2 storage and training.

use serde::{Deserialize, Serialize};

use crate::p2p::types::P2pConfig;
use crate::types::TrainingParams;

/// R2/S3-compatible storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub endpoint: String,
    pub bucket: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:9000".to_string(),
            bucket: crate::paths::DEFAULT_BUCKET.to_string(),
            access_key_id: "minioadmin".to_string(),
            secret_access_key: "minioadmin".to_string(),
            region: "auto".to_string(),
        }
    }
}

/// Coordinator server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    pub host: String,
    pub port: u16,
    pub storage: StorageConfig,
    pub min_contributions: u64,
    pub staleness_decay: f64,
    pub max_staleness: u64,
    pub outer_lr: f64,
    pub outer_momentum: f64,
    pub keep_versions: u64,
    pub vocab_size: u32,
    /// Minimum total contribution weight before a checkpoint can be produced.
    /// Works alongside min_contributions: both must be satisfied.
    /// Default 1.5 — a single strong node (weight ~2.0) can trigger, but a
    /// single weak node (weight 0.5) cannot.
    #[serde(default = "default_min_weight")]
    pub min_weight: f64,
    /// Max storage in GB for coordinator (checkpoints + deltas + optimizer state).
    /// When exceeded, oldest versions are deleted. 0 = use keep_versions only.
    #[serde(default)]
    pub max_storage_gb: u64,
    /// External S3 endpoint returned to nodes via /config.
    /// When set, the /config and /nodes/register endpoints return this instead of storage.endpoint.
    /// Use this when the coordinator uses localhost:9000 for internal MinIO but nodes need
    /// an external URL (e.g. http://PUBLIC_IP:9000).
    #[serde(default)]
    pub external_storage_endpoint: Option<String>,

    /// Enable delta rebasing for stale deltas. Subtracts model drift before merge.
    #[serde(default = "default_enable_rebasing")]
    pub enable_rebasing: bool,
    /// Staleness threshold above which rebasing kicks in.
    #[serde(default = "default_rebasing_threshold")]
    pub rebasing_threshold: u64,
    /// How much drift to subtract (0.0-1.0). 0.5 = conservative, 1.0 = full.
    #[serde(default = "default_rebasing_coefficient")]
    pub rebasing_coefficient: f64,

    /// P2P networking configuration (DHT + gossip).
    #[serde(default)]
    pub p2p: P2pConfig,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            storage: StorageConfig::default(),
            min_contributions: 4,
            staleness_decay: 0.9,
            max_staleness: 10,
            outer_lr: 0.1,
            outer_momentum: 0.9,
            keep_versions: 10,
            vocab_size: 32768,
            min_weight: default_min_weight(),
            max_storage_gb: 0,
            external_storage_endpoint: None,
            enable_rebasing: default_enable_rebasing(),
            rebasing_threshold: default_rebasing_threshold(),
            rebasing_coefficient: default_rebasing_coefficient(),
            p2p: P2pConfig::default(),
        }
    }
}

fn default_enable_rebasing() -> bool { true }
fn default_rebasing_threshold() -> u64 { 3 }
fn default_rebasing_coefficient() -> f64 { 0.5 }

/// Node client configuration (loaded from TOML).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub coordinator_url: String,
    pub api_key: String,
    pub storage: StorageConfig,
    pub gpu_device: i32,
    pub target_push_interval_secs: f64,
    pub min_inner_steps: u64,
    pub max_inner_steps: u64,
    pub cache_dir: String,
    pub max_cache_gb: u64,
    /// Batch size. If not set, auto-detected from VRAM.
    /// Set via toml `batch_size = N` or env `BATCH_SIZE=N`.
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,
    #[serde(default)]
    pub training_params: Option<TrainingParams>,
    /// Maximum fraction of available RAM to use (0.0-1.0). Default 0.80.
    #[serde(default = "default_max_memory_fraction")]
    pub max_memory_fraction: f64,
    /// Alias for batch_size (backwards compat).
    #[serde(default)]
    pub force_batch_size: Option<usize>,
    /// Force CPU backend (set via CLI --cpu flag, not from config file)
    #[serde(skip)]
    pub force_cpu: bool,

    // ── Ablation config (v0.2) ──────────────────────────────────
    /// Compression pipeline: "block" (v0.2 row-level) or "unstructured" (v0.1 element-level).
    #[serde(default = "default_compression_pipeline")]
    pub compression_pipeline: String,
    /// Fixed retention fraction. If set, overrides adaptive top-k.
    #[serde(default)]
    pub compression_retention: Option<f32>,
    /// Quantization mode: "int8_block" (per-row), "int8_tensor" (per-tensor), "bf16" (none).
    #[serde(default = "default_quantization_mode")]
    pub quantization_mode: String,
    /// Enable importance-weighted selection for top-k/block.
    #[serde(default)]
    pub use_importance: bool,

    /// P2P networking configuration (DHT + gossip).
    #[serde(default)]
    pub p2p: P2pConfig,
}

fn default_min_weight() -> f64 {
    1.5
}

fn default_max_memory_fraction() -> f64 {
    0.80
}

fn default_seq_len() -> usize {
    512
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            coordinator_url: "http://localhost:8000".to_string(),
            api_key: String::new(),
            storage: StorageConfig::default(),
            gpu_device: -1,
            target_push_interval_secs: 60.0,
            min_inner_steps: 10,
            max_inner_steps: 500,
            cache_dir: "~/.distrain/cache".to_string(),
            max_cache_gb: 100,
            batch_size: None,
            seq_len: 512,
            training_params: None,
            max_memory_fraction: default_max_memory_fraction(),
            force_batch_size: None,
            force_cpu: false,
            compression_pipeline: default_compression_pipeline(),
            compression_retention: None,
            quantization_mode: default_quantization_mode(),
            use_importance: false,
            p2p: P2pConfig::default(),
        }
    }
}

fn default_compression_pipeline() -> String { "unstructured".to_string() }
fn default_quantization_mode() -> String { "int8_tensor".to_string() }
