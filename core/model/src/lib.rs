//! DistrainTransformer in Burn — pure Rust, no Python dependency.
//!
//! This is a faithful port of the Python model architecture:
//! - GQA (Grouped Query Attention) with RoPE
//! - SwiGLU FFN
//! - RMSNorm (pre-norm)
//! - Tied embeddings with learnable output scaling
//! - DeepSeek-style per-layer initialization
//!
//! The same model runs on every platform: CPU (ndarray), GPU (wgpu/tch),
//! Android (NDK), iOS, and Browser (WASM + WebGPU).

pub mod config;
pub mod model;
pub mod checkpoint;
pub mod compression;
pub mod importance;
pub mod lowrank;
pub mod training;

pub use config::{ModelConfig, ModelPreset};
pub use model::{DistrainTransformer, DistrainTransformerModule, compute_lm_loss, precompute_rope_tables};
pub use training::compute_shard_assignment;

// ── Centralized backend types ────────────────────────────────────────
// GPU preferred everywhere (Metal/Vulkan/WebGPU), CPU fallback.
// All node crates (CLI, desktop, WASM, mobile FFI) use these types.

use burn::backend::Autodiff;
use burn_wgpu::Wgpu;
use burn_ndarray::NdArray;

/// GPU via wgpu (Metal on macOS, Vulkan on Linux/Windows, WebGPU in browser).
pub type WgpuBackend = Autodiff<Wgpu>;
pub type WgpuDevice = <Wgpu as burn::tensor::backend::Backend>::Device;

/// When compiled with CUDA, GpuBackend uses CUDA. Otherwise wgpu.
#[cfg(feature = "cuda")]
pub type GpuBackend = Autodiff<burn_cuda::Cuda>;
#[cfg(feature = "cuda")]
pub type GpuDevice = <burn_cuda::Cuda as burn::tensor::backend::Backend>::Device;

#[cfg(not(feature = "cuda"))]
pub type GpuBackend = WgpuBackend;
#[cfg(not(feature = "cuda"))]
pub type GpuDevice = WgpuDevice;

/// CPU via ndarray. Used when no GPU is available or in tests.
pub type CpuBackend = Autodiff<NdArray<f32>>;
pub type CpuDevice = <NdArray<f32> as burn::tensor::backend::Backend>::Device;
