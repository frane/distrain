//! Checkpoint loading: safetensors → Burn model parameters.
//!
//! Handles the cross-framework interchange format:
//! - Python (PyTorch) saves safetensors with specific key naming
//! - This module maps those keys to Burn's parameter structure
//! - Enables Burn nodes to load checkpoints produced by the Python aggregator

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;

/// Key mapping from Python model parameter names to Burn model parameter names.
///
/// Python keys look like: `layers.0.attention.q_proj.weight`
/// Burn keys look like:   `layers.0.attention.q_proj.weight`
///
/// Since we use the same names, most mappings are identity.
/// The main difference is the embedding key naming.
pub fn python_to_burn_key(python_key: &str) -> String {
    // Most keys map directly — we used the same naming convention
    python_key.to_string()
}

/// Load a safetensors file and return a map of parameter name → tensor data.
pub fn load_safetensors_map(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let data = std::fs::read(path).context("Failed to read safetensors file")?;
    let tensors = SafeTensors::deserialize(&data).context("Failed to parse safetensors")?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let burn_key = python_to_burn_key(&name);
        let float_data = tensor_view_to_f32(&view)?;
        result.insert(burn_key, float_data);
    }

    Ok(result)
}

/// Convert safetensors tensor view bytes to Vec<f32>.
fn tensor_view_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::BF16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::F16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect();
            Ok(floats)
        }
        other => anyhow::bail!("Unsupported dtype: {:?}", other),
    }
}

/// Convert BF16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert IEEE 754 FP16 to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut e = 0i32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let exponent_f32 = (127 - 15 + 1 + e) as u32;
        let mantissa_f32 = (m & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (exponent_f32 << 23) | mantissa_f32);
    }
    if exponent == 0x1F {
        // Inf / NaN
        let mantissa_f32 = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | mantissa_f32);
    }

    let exponent_f32 = exponent - 15 + 127;
    let mantissa_f32 = mantissa << 13;
    f32::from_bits((sign << 31) | (exponent_f32 << 23) | mantissa_f32)
}

/// Load shapes from a safetensors file without loading full tensor data.
pub fn load_safetensors_shapes(path: &Path) -> Result<HashMap<String, Vec<usize>>> {
    let data = std::fs::read(path).context("Failed to read safetensors file")?;
    let tensors = SafeTensors::deserialize(&data).context("Failed to parse safetensors")?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let burn_key = python_to_burn_key(&name);
        result.insert(burn_key, view.shape().to_vec());
    }

    Ok(result)
}

/// Load shapes from in-memory safetensors bytes.
pub fn load_safetensors_shapes_from_bytes(data: &[u8]) -> Result<HashMap<String, Vec<usize>>> {
    let tensors = SafeTensors::deserialize(data).context("Failed to parse safetensors")?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let burn_key = python_to_burn_key(&name);
        result.insert(burn_key, view.shape().to_vec());
    }

    Ok(result)
}

/// Load a safetensors file from in-memory bytes and return a map of parameter name → tensor data.
pub fn load_safetensors_map_from_bytes(data: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    let tensors = SafeTensors::deserialize(data).context("Failed to parse safetensors")?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let burn_key = python_to_burn_key(&name);
        let float_data = tensor_view_to_f32(&view)?;
        result.insert(burn_key, float_data);
    }

    Ok(result)
}

/// Serialize model parameters to safetensors bytes in memory (no file I/O).
pub fn save_state_dict_safetensors_bytes(
    state_dict: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<u8>> {
    use safetensors::tensor::Dtype;

    let mut tensors: Vec<(&str, safetensors::tensor::TensorView<'_>)> = Vec::new();
    let mut buffers: Vec<Vec<u8>> = Vec::new();

    // Pre-allocate byte buffers
    for data in state_dict.values() {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        buffers.push(bytes);
    }

    // Create tensor views referencing the buffers
    let names: Vec<&str> = state_dict.keys().map(|s| s.as_str()).collect();
    for (i, name) in names.iter().enumerate() {
        let shape = shapes.get(*name).context("Missing shape")?;
        let view = safetensors::tensor::TensorView::new(
            Dtype::F32,
            shape.clone(),
            &buffers[i],
        )?;
        tensors.push((name, view));
    }

    safetensors::serialize(tensors, &None).context("Failed to serialize safetensors to bytes")
}

/// Save model parameters as safetensors (for checkpoint upload).
pub fn save_state_dict_safetensors(
    state_dict: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    path: &Path,
) -> Result<()> {
    use safetensors::tensor::Dtype;

    let mut tensors: Vec<(&str, safetensors::tensor::TensorView<'_>)> = Vec::new();
    let mut buffers: Vec<Vec<u8>> = Vec::new();

    // Pre-allocate byte buffers
    for (_name, data) in state_dict {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        buffers.push(bytes);
    }

    // Create tensor views referencing the buffers
    let names: Vec<&str> = state_dict.keys().map(|s| s.as_str()).collect();
    for (i, name) in names.iter().enumerate() {
        let shape = shapes.get(*name).context("Missing shape")?;
        let view = safetensors::tensor::TensorView::new(
            Dtype::F32,
            shape.clone(),
            &buffers[i],
        )?;
        tensors.push((name, view));
    }

    safetensors::serialize_to_file(tensors, &None, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        // BF16 representation of 1.0: sign=0, exp=127 (0x3F80 >> 7), mantissa=0
        // BF16 for 1.0 = 0x3F80
        let result = bf16_to_f32(0x3F80);
        assert!((result - 1.0).abs() < 1e-6, "Got {result}");
    }

    #[test]
    fn test_bf16_negative() {
        // BF16 for -2.0 = 0xC000
        let result = bf16_to_f32(0xC000);
        assert!((result - (-2.0)).abs() < 1e-6, "Got {result}");
    }

    #[test]
    fn test_f16_conversion() {
        // FP16 for 1.0 = 0x3C00
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 1e-6, "Got {result}");
    }

    #[test]
    fn test_safetensors_bytes_roundtrip() {
        let mut state = HashMap::new();
        state.insert("a".to_string(), vec![1.0f32, 2.0, 3.0]);
        state.insert("b".to_string(), vec![4.0f32, 5.0]);

        let mut shapes = HashMap::new();
        shapes.insert("a".to_string(), vec![3usize]);
        shapes.insert("b".to_string(), vec![2usize]);

        let bytes = save_state_dict_safetensors_bytes(&state, &shapes).unwrap();
        assert!(!bytes.is_empty());

        let loaded = load_safetensors_map_from_bytes(&bytes).unwrap();
        assert_eq!(loaded["a"], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded["b"], vec![4.0, 5.0]);
    }

    #[test]
    fn test_multidim_shapes_roundtrip() {
        // Simulate embedding weight [256, 64] = 16384 floats
        let data: Vec<f32> = (0..16384).map(|i| i as f32 * 0.001).collect();
        let mut state = HashMap::new();
        state.insert("embedding.weight".to_string(), data);
        state.insert("norm.weight".to_string(), vec![1.0f32; 64]);

        let mut shapes = HashMap::new();
        shapes.insert("embedding.weight".to_string(), vec![256usize, 64]);
        shapes.insert("norm.weight".to_string(), vec![64usize]);

        let bytes = save_state_dict_safetensors_bytes(&state, &shapes).unwrap();

        // Verify shapes are preserved
        let loaded_shapes = load_safetensors_shapes_from_bytes(&bytes).unwrap();
        assert_eq!(loaded_shapes["embedding.weight"], vec![256, 64]);
        assert_eq!(loaded_shapes["norm.weight"], vec![64]);

        // Verify data is preserved
        let loaded = load_safetensors_map_from_bytes(&bytes).unwrap();
        assert_eq!(loaded["embedding.weight"].len(), 16384);
        assert_eq!(loaded["norm.weight"].len(), 64);
    }
}
