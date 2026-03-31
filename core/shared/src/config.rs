//! Configuration types for R2 storage and training.

use serde::{Deserialize, Serialize};

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
        }
    }
}

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
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,
    #[serde(default)]
    pub training_params: Option<TrainingParams>,
    /// Maximum fraction of available RAM to use (0.0-1.0). Default 0.80.
    #[serde(default = "default_max_memory_fraction")]
    pub max_memory_fraction: f64,
    /// Force batch_size without probing. Skips GPU subprocess calibration entirely.
    #[serde(default)]
    pub force_batch_size: Option<usize>,
    /// Force CPU backend (set via CLI --cpu flag, not from config file)
    #[serde(skip)]
    pub force_cpu: bool,
}

fn default_min_weight() -> f64 {
    1.5
}

fn default_max_memory_fraction() -> f64 {
    0.80
}

fn default_batch_size() -> usize {
    4
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
            batch_size: 4,
            seq_len: 512,
            training_params: None,
            max_memory_fraction: default_max_memory_fraction(),
            force_batch_size: None,
            force_cpu: false,
        }
    }
}
