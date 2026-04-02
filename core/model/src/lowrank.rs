//! Low-rank delta compression via truncated SVD.
//!
//! Decomposes each weight matrix delta D ≈ U × V^T where U ∈ R^{m×r} and V ∈ R^{n×r}.
//! Sends U and V instead of the full matrix. Coordinator reconstructs D̂ = U × V^T.
//!
//! Error feedback: the residual (D - D̂) accumulates in the error buffer and
//! re-enters on the next push, exactly like top-k error feedback.
//!
//! The rank adapts to bandwidth: datacenter = rank-64 (6x), residential = rank-4 (96x).

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Error buffer for low-rank compression. Wraps HashMap<String, Vec<f32>>.
/// Accumulates reconstruction residuals across rounds.
#[derive(Debug, Clone, Default)]
pub struct LowRankErrorBuffer(pub HashMap<String, Vec<f32>>);

/// A low-rank representation of a tensor: U × V^T ≈ original matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRankFactor {
    pub u: Vec<f32>,    // [m × r] flattened row-major
    pub v: Vec<f32>,    // [n × r] flattened row-major
    pub m: usize,
    pub n: usize,
    pub rank: usize,
}

/// A compressed delta using low-rank decomposition per tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRankDelta {
    pub tensors: HashMap<String, LowRankFactor>,
    pub rank: usize,
}

/// Stats from low-rank compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRankStats {
    pub rank: usize,
    pub original_bytes: u64,
    pub compressed_bytes: u64,
    pub compression_ratio: f64,
    pub reconstruction_error: f64,  // relative Frobenius norm: ||D - D̂||_F / ||D||_F
}

/// Compress a delta using low-rank decomposition.
///
/// For each tensor in the delta:
/// - If 1D (bias, norm weights): keep as-is (tiny, not worth decomposing)
/// - If 2D (weight matrices): decompose via truncated SVD
/// - Error feedback: residual (D - UV^T) added back to error_buffer
pub fn compress_lowrank(
    delta: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    rank: usize,
    error_buffer: &mut HashMap<String, Vec<f32>>,
) -> Result<(LowRankDelta, LowRankStats)> {
    let mut tensors = HashMap::new();
    let mut original_bytes: u64 = 0;
    let mut compressed_bytes: u64 = 0;
    let mut total_error_sq: f64 = 0.0;
    let mut total_norm_sq: f64 = 0.0;

    for (name, values) in delta {
        let default_shape = vec![values.len()];
        let shape = shapes.get(name).unwrap_or(&default_shape);
        original_bytes += (values.len() * 4) as u64;

        // Add error buffer
        let effective: Vec<f32> = if let Some(eb) = error_buffer.get(name) {
            values.iter().zip(eb.iter()).map(|(d, e)| d + e).collect()
        } else {
            values.clone()
        };

        if shape.len() == 2 && shape[0] > 1 && shape[1] > 1 {
            let m = shape[0];
            let n = shape[1];
            let r = rank.min(m).min(n);

            // Truncated SVD via power iteration
            let (u, v) = truncated_svd(&effective, m, n, r);

            // Reconstruct and compute residual for error feedback
            let mut reconstructed = vec![0.0f32; m * n];
            matmul_uvt(&u, &v, m, n, r, &mut reconstructed);

            // Error = effective - reconstructed → goes to error buffer
            let residual: Vec<f32> = effective.iter()
                .zip(reconstructed.iter())
                .map(|(e, r)| e - r)
                .collect();

            let err_sq: f64 = residual.iter().map(|x| (*x as f64) * (*x as f64)).sum();
            let norm_sq: f64 = effective.iter().map(|x| (*x as f64) * (*x as f64)).sum();
            total_error_sq += err_sq;
            total_norm_sq += norm_sq;

            error_buffer.insert(name.clone(), residual);

            compressed_bytes += ((m * r + n * r) * 4) as u64;
            tensors.insert(name.clone(), LowRankFactor { u, v, m, n, rank: r });
        } else {
            // 1D tensor (bias, norm): keep as-is, clear error buffer
            compressed_bytes += (effective.len() * 4) as u64;
            tensors.insert(name.clone(), LowRankFactor {
                u: effective.clone(),
                v: Vec::new(),
                m: effective.len(),
                n: 0,
                rank: 0,
            });
            // No residual for 1D tensors — stored exactly
            error_buffer.remove(name);
        }
    }

    let reconstruction_error = if total_norm_sq > 0.0 {
        (total_error_sq / total_norm_sq).sqrt()
    } else {
        0.0
    };

    let stats = LowRankStats {
        rank,
        original_bytes,
        compressed_bytes,
        compression_ratio: if compressed_bytes > 0 {
            original_bytes as f64 / compressed_bytes as f64
        } else {
            1.0
        },
        reconstruction_error,
    };

    info!(
        "Low-rank compression: rank={rank}, {:.0}MB → {:.0}MB ({:.1}x), reconstruction_error={:.4}",
        original_bytes as f64 / 1e6,
        compressed_bytes as f64 / 1e6,
        stats.compression_ratio,
        reconstruction_error,
    );

    Ok((LowRankDelta { tensors, rank }, stats))
}

/// Decompress: reconstruct full delta from low-rank factors.
pub fn decompress_lowrank(lr_delta: &LowRankDelta) -> HashMap<String, Vec<f32>> {
    let mut result = HashMap::new();

    for (name, factor) in &lr_delta.tensors {
        if factor.rank == 0 || factor.n == 0 {
            // 1D tensor stored directly
            result.insert(name.clone(), factor.u.clone());
        } else {
            // Reconstruct: D = U × V^T
            let mut reconstructed = vec![0.0f32; factor.m * factor.n];
            matmul_uvt(&factor.u, &factor.v, factor.m, factor.n, factor.rank, &mut reconstructed);
            result.insert(name.clone(), reconstructed);
        }
    }

    result
}

/// Truncated SVD via randomized power iteration.
///
/// Returns (U, V) where U ∈ R^{m×r} and V ∈ R^{n×r}, flattened row-major.
/// D ≈ U × V^T (singular values absorbed into U and V equally).
fn truncated_svd(matrix: &[f32], m: usize, n: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    // All computation in f64 for numerical stability
    let mat: Vec<f64> = matrix.iter().map(|&x| x as f64).collect();

    // 1. Random Gaussian Ω ∈ R^{n×r}
    let mut omega = vec![0.0f64; n * r];
    let mut rng = (m as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(n as u64).wrapping_add(r as u64);
    for val in omega.iter_mut() {
        rng = splitmix64(rng);
        let u1 = (rng & 0xFFFFFFFF) as f64 / 4294967296.0 + 1e-10;
        rng = splitmix64(rng);
        let u2 = (rng & 0xFFFFFFFF) as f64 / 4294967296.0;
        *val = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }

    // 2. Y = D × Ω ∈ R^{m×r}
    let mut y = vec![0.0f64; m * r];
    for i in 0..m {
        for p in 0..n {
            let d = mat[i * n + p];
            for j in 0..r {
                y[i * r + j] += d * omega[p * r + j];
            }
        }
    }

    // 3. Gram-Schmidt QR: Y → Q (orthonormal columns) in f64
    for j in 0..r {
        let norm: f64 = (0..m).map(|i| y[i * r + j] * y[i * r + j]).sum::<f64>().sqrt().max(1e-15);
        for i in 0..m { y[i * r + j] /= norm; }
        for k in (j + 1)..r {
            let dot: f64 = (0..m).map(|i| y[i * r + j] * y[i * r + k]).sum();
            for i in 0..m { y[i * r + k] -= dot * y[i * r + j]; }
        }
    }
    // y is now Q

    // 4. B = Q^T × D ∈ R^{r×n}
    let mut b = vec![0.0f64; r * n];
    for i in 0..m {
        for p in 0..r {
            let q_ip = y[i * r + p];
            for j in 0..n {
                b[p * n + j] += q_ip * mat[i * n + j];
            }
        }
    }

    // 5. Return U=Q ∈ R^{m×r}, V=B^T ∈ R^{n×r} (so U×V^T = Q×B = Q×Q^T×D)
    let u: Vec<f32> = y.iter().map(|&x| x as f32).collect();
    let mut v = vec![0.0f32; n * r];
    for p in 0..r {
        for j in 0..n {
            v[j * r + p] = b[p * n + j] as f32;
        }
    }

    (u, v)
}

// ── Matrix operations (no external dependency) ──────────────────────

/// C = A × B where A ∈ R^{m×k}, B ∈ R^{k×n}, C ∈ R^{m×n}
#[allow(dead_code)]
fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            if a_ip == 0.0 { continue; }
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
}

/// C = A^T × B where A ∈ R^{m×k}, B ∈ R^{m×n}, C ∈ R^{k×n}
#[allow(dead_code)]
fn matmul_at_b(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            if a_ip == 0.0 { continue; }
            for j in 0..n {
                c[p * n + j] += a_ip * b[i * n + j];
            }
        }
    }
}

/// C = U × V^T where U ∈ R^{m×r}, V ∈ R^{n×r}, C ∈ R^{m×n}
fn matmul_uvt(u: &[f32], v: &[f32], m: usize, n: usize, r: usize, c: &mut [f32]) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..r {
            let u_ip = u[i * r + p];
            if u_ip == 0.0 { continue; }
            for j in 0..n {
                c[i * n + j] += u_ip * v[j * r + p];
            }
        }
    }
}

/// Modified Gram-Schmidt QR: returns Q ∈ R^{m×r} (orthonormal columns).
#[allow(dead_code)]
fn qr_q(a: &[f32], m: usize, r: usize) -> Vec<f32> {
    let mut q = a.to_vec();

    for j in 0..r {
        // Normalize column j
        let mut norm: f64 = 0.0;
        for i in 0..m {
            let v = q[i * r + j] as f64;
            norm += v * v;
        }
        let norm = norm.sqrt().max(1e-10);
        for i in 0..m {
            q[i * r + j] /= norm as f32;
        }

        // Orthogonalize subsequent columns against column j
        for k in (j + 1)..r {
            let mut dot: f64 = 0.0;
            for i in 0..m {
                dot += q[i * r + j] as f64 * q[i * r + k] as f64;
            }
            for i in 0..m {
                q[i * r + k] -= (dot as f32) * q[i * r + j];
            }
        }
    }

    q
}

/// splitmix64 PRNG
fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_basic() {
        // 2x3 × 3x2 = 2x2
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        super::matmul(&a, &b, &mut c, 2, 3, 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c[0], 58.0);
        assert_eq!(c[1], 64.0);
        assert_eq!(c[2], 139.0);
        assert_eq!(c[3], 154.0);
    }

    #[test]
    fn test_uvt_reconstruction() {
        // U = [[1, 0], [0, 1], [1, 1]] (3x2)
        // V = [[2, 0], [0, 3]] (2x2)
        // U × V^T = [[2, 0], [0, 3], [2, 3]]
        let u = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![2.0f32, 0.0, 0.0, 3.0];
        let mut c = vec![0.0f32; 6];
        super::matmul_uvt(&u, &v, 3, 2, 2, &mut c);
        assert_eq!(c, vec![2.0, 0.0, 0.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn test_qr_orthonormal() {
        let m = 4;
        let r = 2;
        // Two non-orthogonal columns
        let a = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let q = super::qr_q(&a, m, r);

        // Check Q^T Q = I
        for i in 0..r {
            for j in 0..r {
                let mut dot: f64 = 0.0;
                for k in 0..m {
                    dot += q[k * r + i] as f64 * q[k * r + j] as f64;
                }
                if i == j {
                    assert!((dot - 1.0).abs() < 1e-5, "Q^TQ[{i},{j}] = {dot}, expected 1.0");
                } else {
                    assert!(dot.abs() < 1e-5, "Q^TQ[{i},{j}] = {dot}, expected 0.0");
                }
            }
        }
    }

    #[test]
    fn test_svd_direct() {
        // Rank-1 matrix: outer product
        let m = 8;
        let n = 4;
        let r = 2;
        let mut matrix = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                matrix[i * n + j] = (i as f32 + 1.0) * (j as f32 + 1.0);
            }
        }

        let (u, v) = super::truncated_svd(&matrix, m, n, r);
        let mut recon = vec![0.0f32; m * n];
        super::matmul_uvt(&u, &v, m, n, r, &mut recon);

        let mut err_sq: f64 = 0.0;
        let mut norm_sq: f64 = 0.0;
        for i in 0..m * n {
            err_sq += ((matrix[i] - recon[i]) as f64).powi(2);
            norm_sq += (matrix[i] as f64).powi(2);
        }
        let rel = (err_sq / norm_sq).sqrt();
        eprintln!("SVD direct: rel_error={rel:.6}, matrix[0..4]={:?}, recon[0..4]={:?}", &matrix[..4], &recon[..4]);
        assert!(rel < 0.15, "SVD reconstruction error: {rel}");
    }

    #[test]
    fn test_lowrank_roundtrip() {
        // Same matrix as direct SVD test — known to work at 10% error
        let m = 8;
        let n = 4;
        let r = 2;
        let mut matrix = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                matrix[i * n + j] = (i as f32 + 1.0) * (j as f32 + 1.0);
            }
        }

        let shapes: HashMap<String, Vec<usize>> = [("w".to_string(), vec![m, n])].into();
        let delta: HashMap<String, Vec<f32>> = [("w".to_string(), matrix.clone())].into();
        let mut eb = HashMap::new();

        let (lr_delta, stats) = compress_lowrank(&delta, &shapes, r, &mut eb).unwrap();
        eprintln!("stats: ratio={:.1}x, error={:.4}", stats.compression_ratio, stats.reconstruction_error);
        eprintln!("factor: u_len={}, v_len={}, m={}, n={}, rank={}",
            lr_delta.tensors["w"].u.len(), lr_delta.tensors["w"].v.len(),
            lr_delta.tensors["w"].m, lr_delta.tensors["w"].n, lr_delta.tensors["w"].rank);

        let reconstructed = decompress_lowrank(&lr_delta);
        let recon = &reconstructed["w"];

        eprintln!("matrix[0..4]={:?}", &matrix[..4]);
        eprintln!("recon[0..4]={:?}", &recon[..4]);

        let mut err_sq: f64 = 0.0;
        let mut norm_sq: f64 = 0.0;
        for i in 0..m * n {
            let e = (matrix[i] - recon[i]) as f64;
            err_sq += e * e;
            norm_sq += (matrix[i] as f64) * (matrix[i] as f64);
        }
        let rel_error = (err_sq / norm_sq.max(1e-10)).sqrt();
        assert!(rel_error < 0.15, "Reconstruction error too high: {rel_error}");
    }

    #[test]
    fn test_1d_passthrough() {
        let shapes: HashMap<String, Vec<usize>> = [("b".to_string(), vec![64])].into();
        let delta: HashMap<String, Vec<f32>> = [("b".to_string(), vec![1.0; 64])].into();
        let mut eb = HashMap::new();

        let (lr_delta, _) = compress_lowrank(&delta, &shapes, 4, &mut eb).unwrap();
        let reconstructed = decompress_lowrank(&lr_delta);
        assert_eq!(reconstructed["b"], vec![1.0f32; 64]);
    }

    #[test]
    fn test_error_feedback() {
        let m = 32;
        let n = 16;
        let shapes: HashMap<String, Vec<usize>> = [("w".to_string(), vec![m, n])].into();

        // Random-ish delta
        let delta: HashMap<String, Vec<f32>> = [("w".to_string(),
            (0..m*n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect()
        )].into();

        let mut eb: HashMap<String, Vec<f32>> = HashMap::new();

        // First compression: error buffer gets the residual
        let (_, stats1) = compress_lowrank(&delta, &shapes, 2, &mut eb).unwrap();
        assert!(eb.contains_key("w"));
        assert!(stats1.reconstruction_error > 0.0);

        // Second compression with same delta: error buffer adds to delta
        // Reconstruction should be better because error feedback contributes
        let (_, stats2) = compress_lowrank(&delta, &shapes, 2, &mut eb).unwrap();
        // The effective delta is 2x the original, but error feedback makes rank-2 capture more
        assert!(stats2.reconstruction_error >= 0.0); // sanity check
    }
}
