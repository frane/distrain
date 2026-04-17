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

/// Metrics from the compression pipeline, for paper instrumentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// L2 norm of delta after error feedback, before top-k sparsification.
    pub dense_norm: f64,
    /// L2 norm of delta after top-k sparsification.
    pub sparse_norm: f64,
    /// sparse_norm / dense_norm. Measures how much gradient signal survives top-k.
    pub retention_ratio: f64,
    pub top_k_fraction: f32,
    pub num_params_total: usize,
    pub num_params_kept: usize,
    pub compressed_bytes: u64,
    pub raw_param_bytes: u64,
}

/// Compression settings. Matches Python `CompressionConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub top_k_fraction: f32,
    pub quantize_int8: bool,
    pub zstd_level: i32,
    pub min_top_k_fraction: f32,
    pub max_top_k_fraction: f32,
    /// Allocate top-k budget per tensor proportionally to gradient norm.
    pub per_tensor_adaptive: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            top_k_fraction: 0.01,
            quantize_int8: true,
            zstd_level: 3,
            min_top_k_fraction: 0.001,
            max_top_k_fraction: 0.1,
            per_tensor_adaptive: true,
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
///
/// When `per_tensor_adaptive` is true, the top-k budget is distributed across
/// tensors proportionally to their L2 norm. Tensors with large gradients get
/// more of the budget; tensors with near-zero gradients may be skipped entirely.
/// The total number of kept elements is approximately the same as uniform k.
pub fn sparsify_topk(
    delta: &HashMap<String, Vec<f32>>,
    k_fraction: f32,
) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<u32>>, HashMap<String, Vec<f32>>) {
    sparsify_topk_impl(delta, k_fraction, false)
}

/// Per-tensor adaptive top-k: allocate budget by gradient norm.
pub fn sparsify_topk_adaptive(
    delta: &HashMap<String, Vec<f32>>,
    k_fraction: f32,
) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<u32>>, HashMap<String, Vec<f32>>) {
    sparsify_topk_impl(delta, k_fraction, true)
}

fn sparsify_topk_impl(
    delta: &HashMap<String, Vec<f32>>,
    k_fraction: f32,
    per_tensor_adaptive: bool,
) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<u32>>, HashMap<String, Vec<f32>>) {
    // Compute per-tensor norms for adaptive budget allocation
    let tensor_norms: HashMap<&str, f64> = if per_tensor_adaptive {
        delta.iter().map(|(name, flat)| {
            let norm = flat.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
            (name.as_str(), norm)
        }).collect()
    } else {
        HashMap::new()
    };
    let total_norm: f64 = tensor_norms.values().sum();
    // Total budget in number of elements
    let total_params: usize = delta.values().map(|v| v.len()).sum();
    let total_budget = (total_params as f64 * k_fraction as f64).ceil() as usize;

    let mut sparse = HashMap::new();
    let mut all_indices = HashMap::new();
    let mut all_values = HashMap::new();

    for (name, flat) in delta {
        let k = if per_tensor_adaptive && total_norm > 0.0 {
            // Allocate budget proportionally to this tensor's norm
            let frac = tensor_norms.get(name.as_str()).copied().unwrap_or(0.0) / total_norm;
            let k = (total_budget as f64 * frac).ceil() as usize;
            k.max(1).min(flat.len())
        } else {
            ((flat.len() as f32 * k_fraction).ceil() as usize).max(1)
        };

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
    let (sparse, indices, values) = if config.per_tensor_adaptive {
        sparsify_topk_adaptive(&delta_with_error, config.top_k_fraction)
    } else {
        sparsify_topk(&delta_with_error, config.top_k_fraction)
    };
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

/// L2 norm of a delta (across all tensors).
fn delta_l2_norm(delta: &HashMap<String, Vec<f32>>) -> f64 {
    delta
        .values()
        .flat_map(|v| v.iter())
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Total element count across all tensors.
fn delta_param_count(delta: &HashMap<String, Vec<f32>>) -> usize {
    delta.values().map(|v| v.len()).sum()
}

/// Non-zero element count across all tensors.
fn delta_nonzero_count(delta: &HashMap<String, Vec<f32>>) -> usize {
    delta
        .values()
        .flat_map(|v| v.iter())
        .filter(|x| **x != 0.0)
        .count()
}

/// Full compression pipeline: error feedback -> top-k -> INT8 -> zstd -> bytes.
/// Returns compressed bytes and instrumentation stats.
#[cfg(feature = "zstd-compression")]
pub fn compress_delta(
    delta: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    config: &CompressionConfig,
    error_buffer: &mut ErrorBuffer,
) -> Result<(Vec<u8>, CompressionStats)> {
    // Step 1: Error feedback
    let delta_with_error = error_buffer.apply(delta);
    let dense_norm = delta_l2_norm(&delta_with_error);
    let num_params_total = delta_param_count(&delta_with_error);
    let raw_param_bytes = (num_params_total * 4) as u64; // f32 = 4 bytes

    // Step 2: Top-k sparsification
    let (sparse, indices, values) = if config.per_tensor_adaptive {
        sparsify_topk_adaptive(&delta_with_error, config.top_k_fraction)
    } else {
        sparsify_topk(&delta_with_error, config.top_k_fraction)
    };
    error_buffer.update(&delta_with_error, &sparse);
    let sparse_norm = delta_l2_norm(&sparse);
    let num_params_kept = delta_nonzero_count(&sparse);

    // Step 3: Optional INT8 quantization
    let (final_values, scales) = if config.quantize_int8 {
        let (q, s) = quantize_values_int8(&values);
        let deq = dequantize_values_int8(&q, &s);
        (deq, Some(s))
    } else {
        (values, None)
    };

    let sparse_delta = SparseDelta {
        indices,
        values: final_values,
        shapes: shapes.clone(),
        scales: scales.map(|s| s.into_iter().collect()),
        config: config.clone(),
    };

    // Step 4: Serialize + zstd
    let json_bytes = sparse_delta_to_json(&sparse_delta)?;
    let compressed = zstd::encode_all(json_bytes.as_slice(), config.zstd_level)
        .context("zstd compression failed")?;
    let compressed_bytes = compressed.len() as u64;

    let retention_ratio = if dense_norm > 0.0 {
        sparse_norm / dense_norm
    } else {
        0.0
    };

    let stats = CompressionStats {
        dense_norm,
        sparse_norm,
        retention_ratio,
        top_k_fraction: config.top_k_fraction,
        num_params_total,
        num_params_kept,
        compressed_bytes,
        raw_param_bytes,
    };

    Ok((compressed, stats))
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
/// Detects format automatically: block-sparse (v0.2), low-rank, or unstructured sparse (v0.1).
#[cfg(feature = "zstd-compression")]
pub fn decompress_delta(data: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    let json_bytes = zstd::decode_all(data).context("zstd decompression failed")?;

    // Try block-sparse format first (v0.2: has "format": "block_v1")
    if let Ok(block) = serde_json::from_slice::<BlockSparseDelta>(&json_bytes) {
        if block.format == "block_v1" {
            return reconstruct_block_dense(&block);
        }
    }

    // Try low-rank format (has "tensors" key with "u","v","m","n","rank")
    if let Ok(lr_delta) = serde_json::from_slice::<crate::lowrank::LowRankDelta>(&json_bytes) {
        if !lr_delta.tensors.is_empty() {
            return Ok(crate::lowrank::decompress_lowrank(&lr_delta));
        }
    }

    // Fall back to unstructured sparse format (v0.1)
    decompress_delta_json(&json_bytes)
}

/// Compress delta using low-rank decomposition + zstd.
#[cfg(feature = "zstd-compression")]
pub fn compress_delta_lowrank(
    delta: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    rank: usize,
    error_buffer: &mut crate::lowrank::LowRankErrorBuffer,
) -> Result<(Vec<u8>, crate::lowrank::LowRankStats)> {
    let (lr_delta, stats) = crate::lowrank::compress_lowrank(delta, shapes, rank, &mut error_buffer.0)?;
    let json_bytes = serde_json::to_vec(&lr_delta).context("Failed to serialize low-rank delta")?;
    let compressed = zstd::encode_all(json_bytes.as_slice(), 3).context("zstd compression failed")?;
    Ok((compressed, stats))
}

// ── Block Sparsity Pipeline ──────────────────────────────────────────
//
// Row-level block selection: keep top k% of rows by L2 norm.
// For 2D weight matrices, operates on rows. For 1D (bias, norm), falls
// back to standard unstructured top-k.
//
// Index encoding: row indices (uint32) instead of individual element
// indices. 32K rows × 4 bytes = 128 KB vs 25M indices × 4 bytes = 100 MB.

/// Block-sparse delta: selected rows from each tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSparseDelta {
    /// Tensor name → selected row indices (sorted).
    pub row_indices: HashMap<String, Vec<u32>>,
    /// Tensor name → flattened values of selected rows.
    /// Length = sum(row_indices[name].len() * cols_for_name).
    pub values: HashMap<String, Vec<f32>>,
    /// Tensor name → original shape.
    pub shapes: HashMap<String, Vec<usize>>,
    /// Tensor name → per-row INT8 scale factors (one per selected row).
    /// Only present if quantize_int8 is enabled.
    pub row_scales: Option<HashMap<String, Vec<f32>>>,
    /// Compression config used.
    pub config: CompressionConfig,
    /// Format marker for auto-detection.
    pub format: String,
}

/// Row-level block sparsification for 2D tensors.
///
/// For each 2D tensor (weight matrix):
///   - Compute L2 norm per row
///   - Keep top k% of rows by norm
///   - Return selected rows with their indices
///
/// For 1D tensors (bias, layer norm):
///   - Fall back to standard unstructured top-k
///
/// Error buffer operates at the row level for 2D tensors.
pub fn sparsify_block(
    delta: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    k_fraction: f32,
) -> (
    HashMap<String, Vec<u32>>,   // row_indices per tensor
    HashMap<String, Vec<f32>>,   // flattened values of selected rows
    HashMap<String, Vec<f32>>,   // sparse reconstruction (dense with zeros for dropped rows)
) {
    let mut all_row_indices = HashMap::new();
    let mut all_values = HashMap::new();
    let mut all_sparse = HashMap::new();

    for (name, flat) in delta {
        let shape = match shapes.get(name) {
            Some(s) => s,
            None => {
                // No shape info — treat as 1D, use unstructured
                let k = ((flat.len() as f32 * k_fraction).ceil() as usize).max(1);
                let (sparse, indices, values) = unstructured_topk_single(flat, k);
                // Store indices as row_indices (element-level for 1D)
                all_row_indices.insert(name.clone(), indices);
                all_values.insert(name.clone(), values);
                all_sparse.insert(name.clone(), sparse);
                continue;
            }
        };

        if shape.len() != 2 {
            // 1D tensor (bias, norm weight) — use unstructured top-k
            let k = ((flat.len() as f32 * k_fraction).ceil() as usize).max(1);
            let (sparse, indices, values) = unstructured_topk_single(flat, k);
            all_row_indices.insert(name.clone(), indices);
            all_values.insert(name.clone(), values);
            all_sparse.insert(name.clone(), sparse);
            continue;
        }

        // 2D tensor — block (row-level) sparsification
        let rows = shape[0];
        let cols = shape[1];
        let k_rows = ((rows as f32 * k_fraction).ceil() as usize).max(1).min(rows);

        // Compute L2 norm per row
        let mut row_norms: Vec<(usize, f64)> = (0..rows)
            .map(|r| {
                let start = r * cols;
                let end = start + cols;
                let norm = flat[start..end]
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>()
                    .sqrt();
                (r, norm)
            })
            .collect();

        // Sort by norm descending, take top k_rows
        row_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected_rows: Vec<usize> = row_norms[..k_rows].iter().map(|(r, _)| *r).collect();
        selected_rows.sort(); // Sort indices for compression

        // Extract selected row values
        let mut values = Vec::with_capacity(k_rows * cols);
        for &r in &selected_rows {
            let start = r * cols;
            let end = start + cols;
            values.extend_from_slice(&flat[start..end]);
        }

        // Build sparse reconstruction
        let mut sparse = vec![0.0f32; flat.len()];
        for &r in &selected_rows {
            let start = r * cols;
            let end = start + cols;
            sparse[start..end].copy_from_slice(&flat[start..end]);
        }

        let indices: Vec<u32> = selected_rows.iter().map(|&r| r as u32).collect();
        all_row_indices.insert(name.clone(), indices);
        all_values.insert(name.clone(), values);
        all_sparse.insert(name.clone(), sparse);
    }

    (all_row_indices, all_values, all_sparse)
}

/// Unstructured top-k for a single flat tensor. Used for 1D fallback in block sparsity.
fn unstructured_topk_single(
    flat: &[f32],
    k: usize,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    let mut indexed: Vec<(usize, f32)> = flat.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    let mut topk: Vec<(usize, f32)> = indexed[..k.min(flat.len())].to_vec();
    topk.sort_by_key(|(idx, _)| *idx);

    let indices: Vec<u32> = topk.iter().map(|(idx, _)| *idx as u32).collect();
    let values: Vec<f32> = topk.iter().map(|(_, val)| *val).collect();
    let mut sparse = vec![0.0f32; flat.len()];
    for &(idx, val) in &topk {
        sparse[idx] = val;
    }
    (sparse, indices, values)
}

/// Per-row INT8 quantization: one scale factor per selected row.
/// More accurate than per-tensor quantization when rows have different magnitudes.
pub fn quantize_values_int8_per_row(
    values: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    row_indices: &HashMap<String, Vec<u32>>,
) -> (HashMap<String, Vec<i8>>, HashMap<String, Vec<f32>>) {
    let mut quantized = HashMap::new();
    let mut all_scales = HashMap::new();

    for (name, vals) in values {
        let shape = shapes.get(name);
        let is_2d = shape.map_or(false, |s| s.len() == 2);

        if is_2d {
            let cols = shape.unwrap()[1];
            let num_rows = row_indices.get(name).map_or(0, |r| r.len());
            let mut q = Vec::with_capacity(vals.len());
            let mut scales = Vec::with_capacity(num_rows);

            for row_idx in 0..num_rows {
                let start = row_idx * cols;
                let end = (start + cols).min(vals.len());
                let row = &vals[start..end];
                let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                scales.push(scale);
                for v in row {
                    q.push((v / scale).round().clamp(-127.0, 127.0) as i8);
                }
            }

            quantized.insert(name.clone(), q);
            all_scales.insert(name.clone(), scales);
        } else {
            // 1D: single scale per tensor (same as before)
            let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let q: Vec<i8> = vals
                .iter()
                .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
                .collect();
            quantized.insert(name.clone(), q);
            all_scales.insert(name.clone(), vec![scale]);
        }
    }

    (quantized, all_scales)
}

/// Dequantize per-row INT8 values.
pub fn dequantize_values_int8_per_row(
    quantized: &HashMap<String, Vec<i8>>,
    row_scales: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    row_indices: &HashMap<String, Vec<u32>>,
) -> HashMap<String, Vec<f32>> {
    let mut result = HashMap::new();

    for (name, vals) in quantized {
        let scales = &row_scales[name];
        let shape = shapes.get(name);
        let is_2d = shape.map_or(false, |s| s.len() == 2);

        if is_2d && scales.len() > 1 {
            let cols = shape.unwrap()[1];
            let num_rows = row_indices.get(name).map_or(0, |r| r.len());
            let mut deq = Vec::with_capacity(vals.len());
            for row_idx in 0..num_rows {
                let start = row_idx * cols;
                let end = (start + cols).min(vals.len());
                let scale = scales[row_idx];
                for v in &vals[start..end] {
                    deq.push(*v as f32 * scale);
                }
            }
            result.insert(name.clone(), deq);
        } else {
            // 1D: single scale
            let scale = scales[0];
            let deq: Vec<f32> = vals.iter().map(|v| *v as f32 * scale).collect();
            result.insert(name.clone(), deq);
        }
    }

    result
}

/// Full block-sparse compression pipeline.
/// Error feedback → block sparsification → per-row INT8 → zstd.
#[cfg(feature = "zstd-compression")]
pub fn compress_delta_block(
    delta: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    config: &CompressionConfig,
    error_buffer: &mut ErrorBuffer,
) -> Result<(Vec<u8>, CompressionStats)> {
    // Step 1: Error feedback
    let delta_with_error = error_buffer.apply(delta);
    let dense_norm = delta_l2_norm(&delta_with_error);
    let num_params_total = delta_param_count(&delta_with_error);
    let raw_param_bytes = (num_params_total * 4) as u64;

    // Step 2: Block sparsification (row-level for 2D, unstructured for 1D)
    let (row_indices, values, sparse) = sparsify_block(&delta_with_error, shapes, config.top_k_fraction);
    error_buffer.update(&delta_with_error, &sparse);
    let sparse_norm = delta_l2_norm(&sparse);
    let num_params_kept = delta_nonzero_count(&sparse);

    // Step 3: Per-row INT8 quantization
    let (final_values, row_scales) = if config.quantize_int8 {
        let (q, s) = quantize_values_int8_per_row(&values, shapes, &row_indices);
        let deq = dequantize_values_int8_per_row(&q, &s, shapes, &row_indices);
        (deq, Some(s))
    } else {
        (values, None)
    };

    let block_delta = BlockSparseDelta {
        row_indices,
        values: final_values,
        shapes: shapes.clone(),
        row_scales,
        config: config.clone(),
        format: "block_v1".to_string(),
    };

    // Step 4: Serialize + zstd
    let json_bytes = serde_json::to_vec(&block_delta)
        .context("Failed to serialize block sparse delta")?;
    let compressed = zstd::encode_all(json_bytes.as_slice(), config.zstd_level)
        .context("zstd compression failed")?;
    let compressed_bytes = compressed.len() as u64;

    let retention_ratio = if dense_norm > 0.0 { sparse_norm / dense_norm } else { 0.0 };

    let stats = CompressionStats {
        dense_norm,
        sparse_norm,
        retention_ratio,
        top_k_fraction: config.top_k_fraction,
        num_params_total,
        num_params_kept,
        compressed_bytes,
        raw_param_bytes,
    };

    Ok((compressed, stats))
}

/// Reconstruct dense tensors from a BlockSparseDelta.
pub fn reconstruct_block_dense(block: &BlockSparseDelta) -> Result<HashMap<String, Vec<f32>>> {
    let mut result = HashMap::new();

    for (name, shape) in &block.shapes {
        let numel: usize = shape.iter().product();
        let mut flat = vec![0.0f32; numel];

        let indices = &block.row_indices[name];
        let values = &block.values[name];
        let is_2d = shape.len() == 2;

        if is_2d {
            let cols = shape[1];
            for (i, &row_idx) in indices.iter().enumerate() {
                let src_start = i * cols;
                let src_end = (src_start + cols).min(values.len());
                let dst_start = row_idx as usize * cols;
                let dst_end = (dst_start + cols).min(flat.len());
                let len = (src_end - src_start).min(dst_end - dst_start);
                flat[dst_start..dst_start + len].copy_from_slice(&values[src_start..src_start + len]);
            }
        } else {
            // 1D: indices are element positions (same as unstructured)
            for (i, &idx) in indices.iter().enumerate() {
                if (idx as usize) < flat.len() && i < values.len() {
                    flat[idx as usize] = values[i];
                }
            }
        }

        result.insert(name.clone(), flat);
    }

    Ok(result)
}

/// Decompress a delta that could be either block-sparse or unstructured format.
pub fn decompress_delta_json_auto(data: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    // Try block format first (has "format": "block_v1" and "row_indices" key)
    if let Ok(block) = serde_json::from_slice::<BlockSparseDelta>(data) {
        if block.format == "block_v1" {
            return reconstruct_block_dense(&block);
        }
    }

    // Fall back to unstructured sparse format
    decompress_delta_json(data)
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
        // top_k_fraction=0.1 on 1000 elements ≈ 100 (±1 due to rounding)
        assert!(nonzero >= 99 && nonzero <= 101, "expected ~100 nonzero, got {nonzero}");
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

        let (compressed, _stats) = compress_delta(&delta, &shapes, &config, &mut buf).unwrap();
        let recovered = decompress_delta(&compressed).unwrap();

        assert_eq!(recovered["weight"].len(), 1000);
        // 10% kept, so ~100 non-zero (±1 due to rounding)
        let nonzero = recovered["weight"].iter().filter(|v| **v != 0.0).count();
        assert!(nonzero >= 99 && nonzero <= 101, "expected ~100 nonzero, got {nonzero}");
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
                "max_top_k_fraction": 0.1,
                "per_tensor_adaptive": false
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

        let (compressed, _stats) = compress_delta(&delta, &shapes, &config, &mut buf).unwrap();

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

        let (compressed, _stats) = compress_delta(&delta, &shapes, &config, &mut buf).unwrap();
        let recovered = decompress_delta(&compressed).unwrap();

        assert_eq!(recovered["param"].len(), 1000);
        let nonzero = recovered["param"].iter().filter(|v| **v != 0.0).count();
        assert!(nonzero >= 99 && nonzero <= 101, "expected ~100 nonzero, got {nonzero}");
    }

    // ── Block Sparsity Tests ──────────────────────────────────────

    fn make_2d_delta(rows: usize, cols: usize) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut delta = HashMap::new();
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 / 100.0).sin()).collect();
        delta.insert("weight".to_string(), data);
        let mut shapes = HashMap::new();
        shapes.insert("weight".to_string(), vec![rows, cols]);
        (delta, shapes)
    }

    #[test]
    fn test_block_sparsify_2d() {
        let (delta, shapes) = make_2d_delta(100, 50);
        let (indices, values, sparse) = sparsify_block(&delta, &shapes, 0.1);

        // Should keep ~10% of rows = 10 rows
        assert_eq!(indices["weight"].len(), 10);
        // Values should have 10 rows × 50 cols = 500 elements
        assert_eq!(values["weight"].len(), 500);
        // Sparse should be same size as original
        assert_eq!(sparse["weight"].len(), 5000);
        // Indices should be sorted
        for i in 1..indices["weight"].len() {
            assert!(indices["weight"][i] > indices["weight"][i - 1], "Row indices not sorted");
        }
    }

    #[test]
    fn test_block_sparsify_1d_fallback() {
        let mut delta = HashMap::new();
        delta.insert("bias".to_string(), vec![1.0f32; 100]);
        let mut shapes = HashMap::new();
        shapes.insert("bias".to_string(), vec![100]);

        let (indices, values, sparse) = sparsify_block(&delta, &shapes, 0.1);

        // 1D should use unstructured top-k: 10% of 100 = 10
        assert_eq!(indices["bias"].len(), 10);
        assert_eq!(values["bias"].len(), 10);
        assert_eq!(sparse["bias"].len(), 100);
    }

    #[test]
    fn test_block_reconstruct_roundtrip() {
        let (delta, shapes) = make_2d_delta(100, 50);
        let (row_indices, values, _) = sparsify_block(&delta, &shapes, 0.5);

        let block = BlockSparseDelta {
            row_indices: row_indices.clone(),
            values,
            shapes: shapes.clone(),
            row_scales: None,
            config: CompressionConfig::default(),
            format: "block_v1".to_string(),
        };

        let reconstructed = reconstruct_block_dense(&block).unwrap();
        assert_eq!(reconstructed["weight"].len(), 5000);

        // Verify selected rows are preserved exactly
        for &r in &row_indices["weight"] {
            let start = r as usize * 50;
            for c in 0..50 {
                assert_eq!(
                    reconstructed["weight"][start + c],
                    delta["weight"][start + c],
                    "Mismatch at row {r}, col {c}"
                );
            }
        }
    }

    #[test]
    fn test_per_row_quantize_roundtrip() {
        let (delta, shapes) = make_2d_delta(10, 50);
        let (row_indices, values, _) = sparsify_block(&delta, &shapes, 0.5);

        let (quantized, scales) = quantize_values_int8_per_row(&values, &shapes, &row_indices);
        let deq = dequantize_values_int8_per_row(&quantized, &scales, &shapes, &row_indices);

        // Check cosine similarity > 0.99
        let orig = &values["weight"];
        let rec = &deq["weight"];
        let dot: f32 = orig.iter().zip(rec.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = rec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            let cos_sim = dot / (norm_a * norm_b);
            assert!(cos_sim > 0.99, "Cosine sim {cos_sim} too low");
        }
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_block_compress_decompress_roundtrip() {
        let (delta, shapes) = make_2d_delta(100, 50);
        let config = CompressionConfig {
            top_k_fraction: 0.2,
            quantize_int8: true,
            ..Default::default()
        };
        let mut buf = ErrorBuffer::new();

        let (compressed, stats) = compress_delta_block(&delta, &shapes, &config, &mut buf).unwrap();
        assert!(stats.compressed_bytes > 0);
        assert!(stats.retention_ratio > 0.0);

        // Decompress (should auto-detect block format)
        let json_bytes = zstd::decode_all(compressed.as_slice()).unwrap();
        let recovered = decompress_delta_json_auto(&json_bytes).unwrap();

        assert_eq!(recovered["weight"].len(), 5000);
        // ~20% of rows kept = 20 rows, ~1000 nonzero values
        let nonzero = recovered["weight"].iter().filter(|v| **v != 0.0).count();
        assert!(nonzero >= 800 && nonzero <= 1200, "expected ~1000 nonzero, got {nonzero}");
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_block_size_vs_unstructured() {
        let (delta, shapes) = make_2d_delta(100, 50);
        let config = CompressionConfig {
            top_k_fraction: 0.2,
            quantize_int8: true,
            ..Default::default()
        };

        let mut buf1 = ErrorBuffer::new();
        let (compressed_unstructured, _) = compress_delta(&delta, &shapes, &config, &mut buf1).unwrap();

        let mut buf2 = ErrorBuffer::new();
        let (compressed_block, _) = compress_delta_block(&delta, &shapes, &config, &mut buf2).unwrap();

        // Block should be smaller due to fewer index bytes
        // (row indices vs individual element indices)
        // Note: for small tensors the difference may be small; for large tensors it's dramatic
        println!(
            "Unstructured: {} bytes, Block: {} bytes",
            compressed_unstructured.len(),
            compressed_block.len()
        );
        // Both should produce valid output
        assert!(compressed_unstructured.len() > 0);
        assert!(compressed_block.len() > 0);
    }

    #[cfg(feature = "zstd-compression")]
    #[test]
    fn test_coordinator_handles_both_formats() {
        let (delta, shapes) = make_2d_delta(100, 50);
        let config = CompressionConfig {
            top_k_fraction: 0.2,
            quantize_int8: false,
            ..Default::default()
        };

        // Compress with old unstructured format
        let mut buf1 = ErrorBuffer::new();
        let (old_compressed, _) = compress_delta(&delta, &shapes, &config, &mut buf1).unwrap();

        // Compress with new block format
        let mut buf2 = ErrorBuffer::new();
        let (new_compressed, _) = compress_delta_block(&delta, &shapes, &config, &mut buf2).unwrap();

        // Coordinator should decompress both
        let old_delta = decompress_delta(&old_compressed).unwrap();
        let new_json = zstd::decode_all(new_compressed.as_slice()).unwrap();
        let new_delta = decompress_delta_json_auto(&new_json).unwrap();

        assert_eq!(old_delta["weight"].len(), 5000);
        assert_eq!(new_delta["weight"].len(), 5000);
    }
}
