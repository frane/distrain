//! Training loop and utilities — runs entirely in Rust via Burn.
//!
//! This replaces the Python `node_train.py` subprocess. The node client
//! calls these functions directly, producing a compressed delta to upload.

use std::collections::HashMap;

use crate::compression::{CompressionConfig, ErrorBuffer, build_sparse_delta, sparse_delta_to_json};
#[cfg(feature = "zstd-compression")]
use crate::compression::compress_delta;

/// Training configuration for a single node push interval.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub inner_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            weight_decay: 0.1,
            grad_clip_norm: 1.0,
            inner_steps: 50,
            batch_size: 4,
            seq_len: 4096,
        }
    }
}

/// Result of a training run — returned to the node client for upload.
#[derive(Debug)]
pub struct TrainingResult {
    /// Compressed delta bytes (ready for upload to R2).
    pub compressed_delta: Vec<u8>,
    /// Final loss value.
    pub final_loss: f32,
    /// Total tokens processed.
    pub total_tokens: usize,
    /// Training duration in seconds.
    pub elapsed_secs: f64,
}

/// Compute outer delta: θ_start - θ_current (parameter-level subtraction).
pub fn compute_outer_delta(
    start_params: &HashMap<String, Vec<f32>>,
    current_params: &HashMap<String, Vec<f32>>,
) -> HashMap<String, Vec<f32>> {
    let mut delta = HashMap::new();
    for (name, start) in start_params {
        let current = &current_params[name];
        let diff: Vec<f32> = start.iter().zip(current.iter()).map(|(s, c)| s - c).collect();
        delta.insert(name.clone(), diff);
    }
    delta
}

/// Compute parameter shapes from the parameter map.
pub fn compute_shapes(params: &HashMap<String, Vec<f32>>) -> HashMap<String, Vec<usize>> {
    params
        .iter()
        .map(|(k, v)| (k.clone(), vec![v.len()]))
        .collect()
}

/// Full training + compression pipeline for a single push interval.
///
/// 1. Snapshot initial parameters
/// 2. Train for `inner_steps`
/// 3. Compute outer delta (θ_start - θ_end)
/// 4. Compress delta with error feedback
/// 5. Return compressed bytes for upload
/// Full training + compression with zstd (native nodes).
#[cfg(feature = "zstd-compression")]
pub fn train_and_compress(
    start_params: &HashMap<String, Vec<f32>>,
    current_params: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    compression_config: &CompressionConfig,
    error_buffer: &mut ErrorBuffer,
) -> anyhow::Result<(Vec<u8>, crate::compression::CompressionStats)> {
    let delta = compute_outer_delta(start_params, current_params);
    compress_delta(&delta, shapes, compression_config, error_buffer)
}

/// Training + compression as JSON (no zstd — for WASM/lightweight nodes).
pub fn train_and_compress_json(
    start_params: &HashMap<String, Vec<f32>>,
    current_params: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    compression_config: &CompressionConfig,
    error_buffer: &mut ErrorBuffer,
) -> anyhow::Result<Vec<u8>> {
    let delta = compute_outer_delta(start_params, current_params);
    let sparse = build_sparse_delta(&delta, shapes, compression_config, error_buffer);
    sparse_delta_to_json(&sparse)
}

/// Nesterov SGD outer optimizer for model deltas (pseudo-gradients).
///
/// Port of Python `NesterovOuterOptimizer` from `diloco.py`.
/// The "gradients" are parameter deltas (θ_start - θ_end) from inner
/// AdamW training, not true gradients. Follows DiLoCo (Douillard et al. 2023).
///
/// Update rule:
///   v = μ·v + δ
///   θ -= η·(μ·v + δ)
pub struct NesterovOuterOptimizer {
    lr: f64,
    momentum: f64,
    velocity: Option<HashMap<String, Vec<f32>>>,
}

impl NesterovOuterOptimizer {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            velocity: None,
        }
    }

    /// Apply Nesterov update: v = μ·v + δ; θ -= η·(μ·v + δ)
    pub fn step(
        &mut self,
        checkpoint: &mut HashMap<String, Vec<f32>>,
        avg_delta: &HashMap<String, Vec<f32>>,
    ) {
        if self.velocity.is_none() {
            self.velocity = Some(
                avg_delta
                    .iter()
                    .map(|(k, v)| (k.clone(), vec![0.0f32; v.len()]))
                    .collect(),
            );
        }

        let velocity = self.velocity.as_mut().unwrap();
        let mu = self.momentum as f32;
        let lr = self.lr as f32;

        for (name, params) in checkpoint.iter_mut() {
            if let Some(delta) = avg_delta.get(name) {
                let vel = velocity.get_mut(name).unwrap();
                for i in 0..params.len() {
                    vel[i] = mu * vel[i] + delta[i];
                    params[i] -= lr * (mu * vel[i] + delta[i]);
                }
            }
        }
    }

    /// Get velocity state for serialization.
    pub fn velocity_state(&self) -> &Option<HashMap<String, Vec<f32>>> {
        &self.velocity
    }

    /// Load velocity state from deserialized data.
    /// Keys should be bare parameter names (without "velocity." prefix).
    pub fn load_velocity(&mut self, state: HashMap<String, Vec<f32>>) {
        self.velocity = Some(state);
    }

    /// Serialize velocity to a state dict with "velocity." prefix (for safetensors).
    pub fn state_dict(&self) -> HashMap<String, Vec<f32>> {
        match &self.velocity {
            Some(v) => v
                .iter()
                .map(|(k, v)| (format!("velocity.{k}"), v.clone()))
                .collect(),
            None => HashMap::new(),
        }
    }

    /// Load from a state dict with "velocity." prefix.
    pub fn load_state_dict(&mut self, state: HashMap<String, Vec<f32>>) {
        let stripped: HashMap<String, Vec<f32>> = state
            .into_iter()
            .filter_map(|(k, v)| k.strip_prefix("velocity.").map(|s| (s.to_string(), v)))
            .collect();
        if !stripped.is_empty() {
            self.velocity = Some(stripped);
        }
    }
}

/// Compute weighted average of multiple deltas.
///
/// Each delta is paired with a weight. Weights are normalized to sum to 1.0,
/// then `avg[k] = Σ(w_i * delta_i[k])` for each parameter key k.
pub fn weighted_average_deltas(
    deltas: &[(HashMap<String, Vec<f32>>, f64)],
) -> Option<HashMap<String, Vec<f32>>> {
    if deltas.is_empty() {
        return None;
    }

    let total_weight: f64 = deltas.iter().map(|(_, w)| w).sum();
    if total_weight == 0.0 {
        return None;
    }

    let mut avg: HashMap<String, Vec<f32>> = HashMap::new();

    for (delta, weight) in deltas {
        let w = (*weight / total_weight) as f32;
        for (k, v) in delta {
            let entry = avg.entry(k.clone()).or_insert_with(|| vec![0.0f32; v.len()]);
            for (i, val) in v.iter().enumerate() {
                entry[i] += w * val;
            }
        }
    }

    Some(avg)
}

/// Deterministic shard assignment — no coordinator needed.
///
/// Every node can independently compute its own shard list AND verify
/// any other node's assignment. Based on hash of (node_id, checkpoint_version).
///
/// Port of Python `compute_shard_assignment` from TASKS.md Task 1.1.
pub fn compute_shard_assignment(
    node_id: &str,
    checkpoint_version: u64,
    total_shards: usize,
    shards_per_node: usize,
) -> Vec<usize> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let count = shards_per_node.min(total_shards);

    // Deterministic seed from node + version
    let mut hasher = DefaultHasher::new();
    format!("{}:{}", checkpoint_version, node_id).hash(&mut hasher);
    let node_seed = hasher.finish();

    // Global shuffle of all shards for this checkpoint version (same for all nodes)
    let mut hasher = DefaultHasher::new();
    checkpoint_version.hash(&mut hasher);
    let global_seed = hasher.finish();

    let mut all_shards: Vec<usize> = (0..total_shards).collect();
    // Fisher-Yates shuffle with deterministic RNG (splitmix64)
    let mut rng_state = global_seed;
    for i in (1..all_shards.len()).rev() {
        rng_state = splitmix64(rng_state);
        let j = (rng_state as usize) % (i + 1);
        all_shards.swap(i, j);
    }

    // Sample `count` shards without replacement using node-specific seed
    // Use reservoir-like selection: shuffle with node seed, take first `count`
    let mut selected = all_shards;
    let mut rng_state = node_seed;
    for i in (1..selected.len()).rev() {
        rng_state = splitmix64(rng_state);
        let j = (rng_state as usize) % (i + 1);
        selected.swap(i, j);
    }
    selected.truncate(count);
    selected
}

/// splitmix64 PRNG — deterministic, fast, portable across platforms.
fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Cosine learning rate schedule with warmup.
pub fn cosine_lr(step: usize, warmup_steps: usize, total_steps: usize, max_lr: f64, min_lr: f64) -> f64 {
    if step < warmup_steps {
        max_lr * (step as f64 / warmup_steps as f64)
    } else {
        let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_outer_delta() {
        let mut start = HashMap::new();
        start.insert("w".to_string(), vec![1.0, 2.0, 3.0]);

        let mut current = HashMap::new();
        current.insert("w".to_string(), vec![0.5, 1.5, 2.5]);

        let delta = compute_outer_delta(&start, &current);
        assert_eq!(delta["w"], vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_cosine_lr_warmup() {
        let lr = cosine_lr(50, 100, 1000, 3e-4, 1e-5);
        assert!((lr - 1.5e-4).abs() < 1e-8, "Got {lr}");
    }

    #[test]
    fn test_cosine_lr_peak() {
        let lr = cosine_lr(100, 100, 1000, 3e-4, 1e-5);
        assert!((lr - 3e-4).abs() < 1e-8, "Got {lr}");
    }

    #[test]
    fn test_cosine_lr_end() {
        let lr = cosine_lr(1000, 100, 1000, 3e-4, 1e-5);
        assert!((lr - 1e-5).abs() < 1e-8, "Got {lr}");
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_train_and_compress() {
        let mut start = HashMap::new();
        start.insert("w".to_string(), vec![1.0; 1000]);

        let mut current = HashMap::new();
        current.insert("w".to_string(), vec![0.9; 1000]);

        let shapes = compute_shapes(&start);
        let config = CompressionConfig {
            top_k_fraction: 0.1,
            quantize_int8: false,
            ..Default::default()
        };
        let mut buf = ErrorBuffer::new();

        let compressed = train_and_compress(&start, &current, &shapes, &config, &mut buf).unwrap();
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_nesterov_optimizer_basic() {
        let mut opt = NesterovOuterOptimizer::new(0.7, 0.9);

        let mut checkpoint = HashMap::new();
        checkpoint.insert("w".to_string(), vec![1.0, 2.0, 3.0]);

        let mut avg_delta = HashMap::new();
        avg_delta.insert("w".to_string(), vec![0.1, 0.2, 0.3]);

        opt.step(&mut checkpoint, &avg_delta);

        // After first step with zero velocity:
        // v = 0.9*0 + [0.1, 0.2, 0.3] = [0.1, 0.2, 0.3]
        // θ -= 0.7 * (0.9*[0.1, 0.2, 0.3] + [0.1, 0.2, 0.3])
        // θ -= 0.7 * [0.19, 0.38, 0.57]
        // θ -= [0.133, 0.266, 0.399]
        assert!((checkpoint["w"][0] - (1.0 - 0.133)).abs() < 0.001);
        assert!((checkpoint["w"][1] - (2.0 - 0.266)).abs() < 0.001);
    }

    #[test]
    fn test_nesterov_state_dict_roundtrip() {
        let mut opt = NesterovOuterOptimizer::new(0.7, 0.9);

        let mut checkpoint = HashMap::new();
        checkpoint.insert("w".to_string(), vec![1.0; 10]);
        let mut delta = HashMap::new();
        delta.insert("w".to_string(), vec![0.1; 10]);

        opt.step(&mut checkpoint, &delta);

        let state = opt.state_dict();
        assert!(state.contains_key("velocity.w"));

        let mut opt2 = NesterovOuterOptimizer::new(0.7, 0.9);
        opt2.load_state_dict(state);
        assert!(opt2.velocity_state().is_some());
        assert_eq!(opt2.velocity_state().as_ref().unwrap()["w"].len(), 10);
    }

    #[test]
    fn test_weighted_average_deltas_basic() {
        let mut d1 = HashMap::new();
        d1.insert("w".to_string(), vec![1.0, 0.0]);
        let mut d2 = HashMap::new();
        d2.insert("w".to_string(), vec![0.0, 1.0]);

        let result = weighted_average_deltas(&[(d1, 1.0), (d2, 1.0)]).unwrap();
        assert!((result["w"][0] - 0.5).abs() < 1e-6);
        assert!((result["w"][1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average_deltas_unequal_weights() {
        let mut d1 = HashMap::new();
        d1.insert("w".to_string(), vec![10.0]);
        let mut d2 = HashMap::new();
        d2.insert("w".to_string(), vec![0.0]);

        let result = weighted_average_deltas(&[(d1, 3.0), (d2, 1.0)]).unwrap();
        assert!((result["w"][0] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_average_deltas_empty() {
        let result = weighted_average_deltas(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_train_and_compress_json() {
        let mut start = HashMap::new();
        start.insert("w".to_string(), vec![1.0; 1000]);

        let mut current = HashMap::new();
        current.insert("w".to_string(), vec![0.9; 1000]);

        let shapes = compute_shapes(&start);
        let config = CompressionConfig {
            top_k_fraction: 0.1,
            quantize_int8: false,
            ..Default::default()
        };
        let mut buf = ErrorBuffer::new();

        let json = train_and_compress_json(&start, &current, &shapes, &config, &mut buf).unwrap();
        assert!(!json.is_empty());
        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert!(parsed["indices"].is_object());
    }

    #[test]
    fn test_shard_assignment_deterministic() {
        let a = compute_shard_assignment("node_a", 5, 1000, 10);
        let b = compute_shard_assignment("node_a", 5, 1000, 10);
        assert_eq!(a, b);
    }

    #[test]
    fn test_shard_assignment_different_nodes() {
        let a = compute_shard_assignment("node_a", 5, 1000, 10);
        let b = compute_shard_assignment("node_b", 5, 1000, 10);
        let set_a: std::collections::HashSet<_> = a.iter().collect();
        let set_b: std::collections::HashSet<_> = b.iter().collect();
        assert_ne!(set_a, set_b);
    }

    #[test]
    fn test_shard_assignment_different_versions() {
        let a = compute_shard_assignment("node_a", 5, 1000, 10);
        let b = compute_shard_assignment("node_a", 6, 1000, 10);
        assert_ne!(a, b);
    }

    #[test]
    fn test_shard_assignment_correct_count() {
        let shards = compute_shard_assignment("node_a", 1, 1000, 10);
        assert_eq!(shards.len(), 10);
        let unique: std::collections::HashSet<_> = shards.iter().collect();
        assert_eq!(unique.len(), 10); // no duplicates
    }

    #[test]
    fn test_shard_assignment_clamps_to_total() {
        let shards = compute_shard_assignment("node_a", 1, 5, 100);
        assert_eq!(shards.len(), 5);
    }
}
