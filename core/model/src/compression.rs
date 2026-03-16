//! Delta compression pipeline in Rust — matches Python `compression.py`.
//!
//! Full pipeline: error feedback → top-k sparsification → INT8 quantization → zstd.
//! This runs on the node side (pure Rust, no Python).
//!
//! The `zstd-compression` feature gates zstd (C library) for platforms where it
//! can't compile (e.g., WASM). Sparsification and quantization are always available.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Compression settings. Matches Python `CompressionConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub top_k_fraction: f32,
    pub quantize_int8: bool,
    pub zstd_level: i32,
    pub min_top_k_fraction: f32,
    pub max_top_k_fraction: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            top_k_fraction: 0.01,
            quantize_int8: true,
            zstd_level: 3,
            min_top_k_fraction: 0.001,
            max_top_k_fraction: 0.1,
        }
    }
}

/// Error buffer: accumulates elements dropped by sparsification.
/// Added back to the next delta so no information is permanently lost.
#[derive(Debug, Clone, Default)]
pub struct ErrorBuffer {
    pub buffer: HashMap<String, Vec<f32>>,
}

impl ErrorBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add error buffer to delta, return combined.
    pub fn apply(&self, delta: &HashMap<String, Vec<f32>>) -> HashMap<String, Vec<f32>> {
        let mut result = HashMap::new();
        for (key, val) in delta {
            if let Some(buf) = self.buffer.get(key) {
                let combined: Vec<f32> = val.iter().zip(buf.iter()).map(|(v, b)| v + b).collect();
                result.insert(key.clone(), combined);
            } else {
                result.insert(key.clone(), val.clone());
            }
        }
        result
    }

    /// Store the residual: what was dropped by sparsification.
    pub fn update(&mut self, delta: &HashMap<String, Vec<f32>>, sparse: &HashMap<String, Vec<f32>>) {
        for (key, d) in delta {
            let s = &sparse[key];
            let residual: Vec<f32> = d.iter().zip(s.iter()).map(|(d, s)| d - s).collect();
            self.buffer.insert(key.clone(), residual);
        }
    }
}

/// Sparse delta representation for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseDelta {
    /// Parameter name → sorted indices of top-k elements.
    pub indices: HashMap<String, Vec<u32>>,
    /// Parameter name → values at those indices (f32 or quantized).
    pub values: HashMap<String, Vec<f32>>,
    /// Parameter name → original tensor shape.
    pub shapes: HashMap<String, Vec<usize>>,
    /// Per-tensor quantization scales (None if not quantized).
    pub scales: Option<HashMap<String, f32>>,
    /// Compression config used.
    pub config: CompressionConfig,
}

/// Top-k sparsification: keep only the largest elements by magnitude.
/// Indices are SORTED — critical for zstd compressibility.
pub fn sparsify_topk(
    delta: &HashMap<String, Vec<f32>>,
    k_fraction: f32,
) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<u32>>, HashMap<String, Vec<f32>>) {
    let mut sparse = HashMap::new();
    let mut all_indices = HashMap::new();
    let mut all_values = HashMap::new();

    for (name, flat) in delta {
        let k = ((flat.len() as f32 * k_fraction).ceil() as usize).max(1);

        // Find top-k by absolute value
        let mut indexed: Vec<(usize, f32)> = flat.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k, then SORT by index (critical for compression)
        let mut topk: Vec<(usize, f32)> = indexed[..k.min(flat.len())].to_vec();
        topk.sort_by_key(|(idx, _)| *idx);

        let indices: Vec<u32> = topk.iter().map(|(idx, _)| *idx as u32).collect();
        let values: Vec<f32> = topk.iter().map(|(_, val)| *val).collect();

        // Reconstruct sparse tensor (dense with zeros for non-top-k)
        let mut sparse_flat = vec![0.0f32; flat.len()];
        for &(idx, val) in &topk {
            sparse_flat[idx] = val;
        }

        sparse.insert(name.clone(), sparse_flat);
        all_indices.insert(name.clone(), indices);
        all_values.insert(name.clone(), values);
    }

    (sparse, all_indices, all_values)
}

/// Per-tensor INT8 quantization.
pub fn quantize_values_int8(
    values: &HashMap<String, Vec<f32>>,
) -> (HashMap<String, Vec<i8>>, HashMap<String, f32>) {
    let mut quantized = HashMap::new();
    let mut scales = HashMap::new();

    for (name, vals) in values {
        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

        let q: Vec<i8> = vals
            .iter()
            .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        quantized.insert(name.clone(), q);
        scales.insert(name.clone(), scale);
    }

    (quantized, scales)
}

/// Reverse INT8 quantization.
pub fn dequantize_values_int8(
    quantized: &HashMap<String, Vec<i8>>,
    scales: &HashMap<String, f32>,
) -> HashMap<String, Vec<f32>> {
    let mut result = HashMap::new();
    for (name, vals) in quantized {
        let scale = scales[name];
        let deq: Vec<f32> = vals.iter().map(|v| *v as f32 * scale).collect();
        result.insert(name.clone(), deq);
    }
    result
}

/// Build a SparseDelta from a delta, applying error feedback and sparsification.
///
/// Always available (no zstd dependency). Returns the SparseDelta struct.
pub fn build_sparse_delta(
    delta: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    config: &CompressionConfig,
    error_buffer: &mut ErrorBuffer,
) -> SparseDelta {
    // Step 1: Error feedback
    let delta_with_error = error_buffer.apply(delta);

    // Step 2: Top-k sparsification
    let (sparse, indices, values) = sparsify_topk(&delta_with_error, config.top_k_fraction);
    error_buffer.update(&delta_with_error, &sparse);

    // Step 3: Optional INT8 quantization
    let (final_values, scales) = if config.quantize_int8 {
        let (q, s) = quantize_values_int8(&values);
        let deq = dequantize_values_int8(&q, &s);
        (deq, Some(s))
    } else {
        (values, None)
    };

    SparseDelta {
        indices,
        values: final_values,
        shapes: shapes.clone(),
        scales: scales.map(|s| s.into_iter().collect()),
        config: config.clone(),
    }
}

/// Serialize a SparseDelta to JSON bytes (no zstd). Always available.
pub fn sparse_delta_to_json(sparse_delta: &SparseDelta) -> Result<Vec<u8>> {
    serde_json::to_vec(sparse_delta).context("Failed to serialize sparse delta")
}

/// Deserialize raw JSON bytes into dense delta tensors. Always available.
pub fn decompress_delta_json(data: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    let sparse_delta: SparseDelta =
        serde_json::from_slice(data).context("Failed to deserialize sparse delta JSON")?;
    reconstruct_dense(&sparse_delta)
}

/// Reconstruct dense tensors from a SparseDelta.
fn reconstruct_dense(sparse_delta: &SparseDelta) -> Result<HashMap<String, Vec<f32>>> {
    let mut result = HashMap::new();
    for (name, shape) in &sparse_delta.shapes {
        let numel: usize = shape.iter().product();
        let mut flat = vec![0.0f32; numel];

        let indices = &sparse_delta.indices[name];
        let values = &sparse_delta.values[name];

        for (i, &idx) in indices.iter().enumerate() {
            flat[idx as usize] = values[i];
        }

        result.insert(name.clone(), flat);
    }
    Ok(result)
}

/// Full compression pipeline: error feedback → top-k → INT8 → zstd → bytes.
#[cfg(feature = "zstd-compression")]
pub fn compress_delta(
    delta: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    config: &CompressionConfig,
    error_buffer: &mut ErrorBuffer,
) -> Result<Vec<u8>> {
    let sparse_delta = build_sparse_delta(delta, shapes, config, error_buffer);
    let json_bytes = sparse_delta_to_json(&sparse_delta)?;
    let compressed = zstd::encode_all(json_bytes.as_slice(), config.zstd_level)
        .context("zstd compression failed")?;
    Ok(compressed)
}

/// Validate a delta for NaN, Inf, and extreme magnitudes.
///
/// Port of Python `validate_delta` from `sanity_checks.py`.
/// Returns Ok(()) if valid, Err(reason) if invalid.
pub fn validate_delta(delta: &HashMap<String, Vec<f32>>) -> std::result::Result<(), String> {
    if delta.is_empty() {
        return Err("Empty delta".to_string());
    }

    for (name, values) in delta {
        for v in values {
            if v.is_nan() {
                return Err(format!("NaN in {name}"));
            }
            if v.is_infinite() {
                return Err(format!("Inf in {name}"));
            }
        }
        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if max_abs > 100.0 {
            return Err(format!(
                "Magnitude too large in {name}: {max_abs:.1}"
            ));
        }
    }

    let total_norm_sq: f64 = delta
        .values()
        .map(|v| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>())
        .sum();
    let total_norm = total_norm_sq.sqrt();

    if total_norm > 1000.0 {
        return Err(format!("Total norm too large: {total_norm:.1}"));
    }
    if total_norm < 1e-10 {
        return Err(format!("Total norm near zero: {total_norm:.2e}"));
    }

    Ok(())
}

/// Decompress bytes → dense delta tensors.
#[cfg(feature = "zstd-compression")]
pub fn decompress_delta(data: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    // Step 1: zstd decompress
    let json_bytes = zstd::decode_all(data).context("zstd decompression failed")?;

    // Step 2: Deserialize and reconstruct
    decompress_delta_json(&json_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_delta(dim: usize, value: f32) -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        m.insert("weight".to_string(), vec![value; dim]);
        m
    }

    fn make_shapes(dim: usize) -> HashMap<String, Vec<usize>> {
        let mut m = HashMap::new();
        m.insert("weight".to_string(), vec![dim]);
        m
    }

    #[test]
    fn test_sparsify_topk_basic() {
        let delta = make_delta(1000, 1.0);
        let (sparse, indices, values) = sparsify_topk(&delta, 0.01);
        assert_eq!(indices["weight"].len(), 10);
        assert_eq!(values["weight"].len(), 10);
        assert_eq!(sparse["weight"].len(), 1000);
    }

    #[test]
    fn test_indices_sorted() {
        use std::collections::HashMap;
        let mut delta = HashMap::new();
        let data: Vec<f32> = (0..10000).map(|i| (i as f32).sin()).collect();
        delta.insert("w".to_string(), data);
        let (_, indices, _) = sparsify_topk(&delta, 0.01);
        let idx = &indices["w"];
        for i in 1..idx.len() {
            assert!(idx[i] > idx[i - 1], "Indices not sorted at {i}");
        }
    }

    #[test]
    fn test_quantize_roundtrip() {
        let mut values = HashMap::new();
        let data: Vec<f32> = (0..100).map(|i| (i as f32 / 10.0).sin()).collect();
        values.insert("w".to_string(), data.clone());

        let (quantized, scales) = quantize_values_int8(&values);
        let deq = dequantize_values_int8(&quantized, &scales);

        // Check cosine similarity > 0.99
        let dot: f32 = data.iter().zip(deq["w"].iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = deq["w"].iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_a * norm_b);
        assert!(cos_sim > 0.99, "Cosine sim {cos_sim} too low");
    }

    #[test]
    fn test_build_sparse_delta_json_roundtrip() {
        let delta = make_delta(1000, 1.0);
        let shapes = make_shapes(1000);
        let config = CompressionConfig {
            top_k_fraction: 0.1,
            quantize_int8: false,
            ..Default::default()
        };
        let mut buf = ErrorBuffer::new();

        let sparse = build_sparse_delta(&delta, &shapes, &config, &mut buf);
        let json_bytes = sparse_delta_to_json(&sparse).unwrap();
        let recovered = decompress_delta_json(&json_bytes).unwrap();

        assert_eq!(recovered["weight"].len(), 1000);
        let nonzero = recovered["weight"].iter().filter(|v| **v != 0.0).count();
        assert_eq!(nonzero, 100);
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_compress_decompress_roundtrip() {
        let delta = make_delta(1000, 1.0);
        let shapes = make_shapes(1000);
        let config = CompressionConfig {
            top_k_fraction: 0.1,
            quantize_int8: false,
            ..Default::default()
        };
        let mut buf = ErrorBuffer::new();

        let compressed = compress_delta(&delta, &shapes, &config, &mut buf).unwrap();
        let recovered = decompress_delta(&compressed).unwrap();

        assert_eq!(recovered["weight"].len(), 1000);
        // 10% kept, so 100 non-zero
        let nonzero = recovered["weight"].iter().filter(|v| **v != 0.0).count();
        assert_eq!(nonzero, 100);
    }

    #[test]
    fn test_validate_delta_valid() {
        let delta = make_delta(100, 0.5);
        assert!(validate_delta(&delta).is_ok());
    }

    #[test]
    fn test_validate_delta_empty() {
        let delta: HashMap<String, Vec<f32>> = HashMap::new();
        assert_eq!(validate_delta(&delta), Err("Empty delta".to_string()));
    }

    #[test]
    fn test_validate_delta_nan() {
        let mut delta = HashMap::new();
        delta.insert("w".to_string(), vec![1.0, f32::NAN, 0.5]);
        assert!(validate_delta(&delta).unwrap_err().contains("NaN"));
    }

    #[test]
    fn test_validate_delta_inf() {
        let mut delta = HashMap::new();
        delta.insert("w".to_string(), vec![1.0, f32::INFINITY]);
        assert!(validate_delta(&delta).unwrap_err().contains("Inf"));
    }

    #[test]
    fn test_validate_delta_magnitude_too_large() {
        let mut delta = HashMap::new();
        delta.insert("w".to_string(), vec![101.0]);
        assert!(validate_delta(&delta).unwrap_err().contains("Magnitude too large"));
    }

    #[test]
    fn test_validate_delta_norm_too_large() {
        // 1001 elements of value 1.0 each → norm = sqrt(1001) ≈ 31.6, that's fine
        // Need total norm > 1000: 1_000_001 elements of 1.0 → norm ≈ 1000
        // Use fewer elements with larger values: 100 elements of 99.0 → norm = 99 * 10 = 990, still fine
        // 102 elements of 99.0 → norm = 99 * sqrt(102) ≈ 999.9, still fine
        // Just use 10001 elements of 10.0 each → norm = 10 * sqrt(10001) ≈ 1000.05
        let mut delta = HashMap::new();
        delta.insert("w".to_string(), vec![10.0; 10001]);
        assert!(validate_delta(&delta).unwrap_err().contains("Total norm too large"));
    }

    #[test]
    fn test_validate_delta_norm_near_zero() {
        let mut delta = HashMap::new();
        delta.insert("w".to_string(), vec![1e-12; 10]);
        assert!(validate_delta(&delta).unwrap_err().contains("near zero"));
    }

    #[test]
    fn test_error_buffer_accumulates() {
        let mut buf = ErrorBuffer::new();
        let delta = make_delta(100, 1.0);
        let shapes = make_shapes(100);
        let config = CompressionConfig {
            top_k_fraction: 0.1,
            quantize_int8: false,
            ..Default::default()
        };

        // First compress (use build_sparse_delta which is always available)
        let _ = build_sparse_delta(&delta, &shapes, &config, &mut buf);
        assert!(!buf.buffer.is_empty());

        // Buffer should have residual for dropped elements
        let residual = &buf.buffer["weight"];
        let nonzero_residual = residual.iter().filter(|v| v.abs() > 1e-8).count();
        assert!(nonzero_residual > 0, "Error buffer should have non-zero residuals");
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_cross_framework_read_python_json() {
        // Simulate what Python compress_delta produces (JSON after zstd decompression)
        let python_json = r#"{
            "indices": {"weight": [0, 5, 10, 50, 99]},
            "values": {"weight": [0.1, -0.3, 0.5, -0.2, 0.4]},
            "shapes": {"weight": [100]},
            "scales": {"weight": 0.003937},
            "config": {
                "top_k_fraction": 0.05,
                "quantize_int8": true,
                "zstd_level": 3,
                "min_top_k_fraction": 0.001,
                "max_top_k_fraction": 0.1
            }
        }"#;

        // Compress with zstd (as Python would)
        let compressed = zstd::encode_all(python_json.as_bytes(), 3).unwrap();

        // Rust should be able to decompress and reconstruct
        let recovered = decompress_delta(&compressed).unwrap();
        assert_eq!(recovered["weight"].len(), 100);
        assert!((recovered["weight"][0] - 0.1).abs() < 1e-6);
        assert!((recovered["weight"][5] - (-0.3)).abs() < 1e-6);
        assert!((recovered["weight"][10] - 0.5).abs() < 1e-6);
        assert_eq!(recovered["weight"][1], 0.0); // not in indices
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_cross_framework_json_schema() {
        // Verify Rust compress_delta produces valid JSON with expected schema
        let delta = make_delta(1000, 1.0);
        let shapes = make_shapes(1000);
        let config = CompressionConfig {
            top_k_fraction: 0.1,
            quantize_int8: true,
            ..Default::default()
        };
        let mut buf = ErrorBuffer::new();

        let compressed = compress_delta(&delta, &shapes, &config, &mut buf).unwrap();

        // Decompress zstd to inspect JSON
        let json_bytes = zstd::decode_all(compressed.as_slice()).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        // Verify top-level keys
        assert!(parsed["indices"].is_object());
        assert!(parsed["values"].is_object());
        assert!(parsed["shapes"].is_object());
        assert!(parsed["config"].is_object());

        // Verify types
        assert!(parsed["indices"]["weight"].is_array());
        assert!(parsed["values"]["weight"].is_array());
        assert!(parsed["shapes"]["weight"].is_array());
        assert!(parsed["config"]["top_k_fraction"].is_f64());
        assert!(parsed["config"]["quantize_int8"].is_boolean());
        assert!(parsed["config"]["zstd_level"].is_i64());

        // Values should be f32 (represented as JSON floats)
        for v in parsed["values"]["weight"].as_array().unwrap() {
            assert!(v.is_f64(), "Expected float value, got {:?}", v);
        }

        // Indices should be ints
        for i in parsed["indices"]["weight"].as_array().unwrap() {
            assert!(i.is_u64(), "Expected uint index, got {:?}", i);
        }
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_cross_framework_roundtrip_with_quantization() {
        // Rust compress with INT8 → inspect JSON → decompress
        let mut delta = HashMap::new();
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();
        delta.insert("param".to_string(), data);
        let mut shapes = HashMap::new();
        shapes.insert("param".to_string(), vec![1000usize]);

        let config = CompressionConfig {
            top_k_fraction: 0.1,
            quantize_int8: true,
            ..Default::default()
        };
        let mut buf = ErrorBuffer::new();

        let compressed = compress_delta(&delta, &shapes, &config, &mut buf).unwrap();
        let recovered = decompress_delta(&compressed).unwrap();

        assert_eq!(recovered["param"].len(), 1000);
        let nonzero = recovered["param"].iter().filter(|v| **v != 0.0).count();
        assert_eq!(nonzero, 100); // 10% top-k
    }
}
