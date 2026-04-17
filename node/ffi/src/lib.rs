#![recursion_limit = "256"]
//! C FFI bindings for the Distrain training engine.
//!
//! Provides a minimal C-compatible API for mobile platforms:
//! - Android (loaded via JNI as .so)
//! - iOS (linked as .a via Swift/ObjC bridge)
//!
//! All functions return JSON strings for cross-language simplicity.
//! The caller is responsible for freeing strings with `distrain_free_string`.
//!
//! Backend selection: prefers GPU (wgpu — Metal on iOS, Vulkan on Android).
//! Falls back to CPU (ndarray) if the GPU probe fails at init time
//! (e.g. older iPad GPUs with incompatible Metal shader support).

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Backend as BurnBackend;
use burn::tensor::ElementConversion;
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use distrain_model::checkpoint::{load_safetensors_map_from_bytes, load_safetensors_shapes_from_bytes};
use distrain_model::compression::{CompressionConfig, ErrorBuffer};
use distrain_model::config::{ModelConfig, ModelPreset};
use distrain_model::model::{compute_lm_loss, precompute_rope_tables, DistrainTransformerModule};
use distrain_model::compression::compress_delta;
use distrain_model::training::{compute_outer_delta, compute_shard_assignment, cosine_lr};
use distrain_model::{GpuBackend, GpuDevice, CpuBackend, CpuDevice};
use distrain_shared::types::TrainingParams;

// --- Runtime backend selection ---

/// Whether GPU probe succeeded. Set once at first init.
static USE_GPU: AtomicBool = AtomicBool::new(true);
static GPU_PROBED: AtomicBool = AtomicBool::new(false);

static GPU_STATE: Mutex<Option<StateInner<GpuBackend>>> = Mutex::new(None);
static CPU_STATE: Mutex<Option<StateInner<CpuBackend>>> = Mutex::new(None);

fn gpu_device() -> GpuDevice {
    #[cfg(target_os = "ios")]
    {
        use burn_wgpu::graphics::Metal;
        use burn_wgpu::{RuntimeOptions, MemoryConfiguration, init_setup, init_device};
        static INIT: std::sync::Once = std::sync::Once::new();
        static mut DEVICE: Option<GpuDevice> = None;
        INIT.call_once(|| {
            let options = RuntimeOptions {
                memory_config: MemoryConfiguration::ExclusivePages,
                ..Default::default()
            };
            let setup = init_setup::<Metal>(&GpuDevice::DefaultDevice, RuntimeOptions {
                memory_config: MemoryConfiguration::ExclusivePages,
                ..Default::default()
            });
            let dev = init_device(setup, options);
            unsafe { DEVICE = Some(dev); }
        });
        unsafe { DEVICE.clone().unwrap_or_default() }
    }
    #[cfg(not(target_os = "ios"))]
    { GpuDevice::default() }
}

fn cpu_device() -> CpuDevice {
    CpuDevice::default()
}

/// Set backend preference. Called from Swift after checking Metal GPU family.
/// `use_gpu`: 1 = GPU (Metal), 0 = CPU (ndarray).
#[no_mangle]
pub extern "C" fn distrain_set_backend(use_gpu: u8) {
    let gpu = use_gpu != 0;
    USE_GPU.store(gpu, Ordering::Relaxed);
    GPU_PROBED.store(true, Ordering::Relaxed);
}

fn using_gpu() -> bool {
    USE_GPU.load(Ordering::Relaxed)
}

// --- Macro for dual-backend dispatch ---
// The body is textually duplicated for each backend arm but monomorphized
// with the correct types by the compiler.

macro_rules! with_state {
    (|$s:ident| $body:expr) => {
        if using_gpu() {
            let mut guard = GPU_STATE.lock().unwrap();
            match guard.as_mut() {
                Some($s) => { $body }
                None => error_cstring("not initialized"),
            }
        } else {
            let mut guard = CPU_STATE.lock().unwrap();
            match guard.as_mut() {
                Some($s) => { $body }
                None => error_cstring("not initialized"),
            }
        }
    };
}

macro_rules! with_state_ref {
    (|$s:ident| $body:expr) => {
        if using_gpu() {
            let guard = GPU_STATE.lock().unwrap();
            match guard.as_ref() {
                Some($s) => { $body }
                None => { let $s: Option<&StateInner<CpuBackend>> = None; let _ = $s; }
            }
        } else {
            let guard = CPU_STATE.lock().unwrap();
            match guard.as_ref() {
                Some($s) => { $body }
                None => { let $s: Option<&StateInner<GpuBackend>> = None; let _ = $s; }
            }
        }
    };
}

// --- Data loader (backend-independent) ---

struct DataLoader {
    shards: Vec<Vec<u16>>,
    batch_size: usize,
    seq_len: usize,
    shard_idx: usize,
    offset: usize,
}

impl DataLoader {
    fn from_files(paths: &[PathBuf], batch_size: usize, seq_len: usize) -> Result<Self, String> {
        let mut shards = Vec::new();
        let mut total: usize = 0;
        for path in paths {
            let bytes = std::fs::read(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            if bytes.len() % 2 != 0 {
                return Err(format!("shard {} has odd byte count", path.display()));
            }
            let tokens: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            total += tokens.len();
            shards.push(tokens);
        }
        if total < batch_size * seq_len {
            return Err(format!(
                "not enough data: {total} tokens < {} needed per batch",
                batch_size * seq_len
            ));
        }
        Ok(Self { shards, batch_size, seq_len, shard_idx: 0, offset: 0 })
    }

    fn next_batch(&mut self) -> Vec<i64> {
        let needed = self.batch_size * self.seq_len;
        let mut result = Vec::with_capacity(needed);
        while result.len() < needed {
            let shard = &self.shards[self.shard_idx];
            let remaining = shard.len() - self.offset;
            let take = (needed - result.len()).min(remaining);
            for &tok in &shard[self.offset..self.offset + take] {
                result.push(tok as i64);
            }
            self.offset += take;
            if self.offset >= shard.len() {
                self.shard_idx = (self.shard_idx + 1) % self.shards.len();
                self.offset = 0;
            }
        }
        result
    }
}

// --- Generic training state ---

struct StateInner<B: BurnBackend> {
    module: DistrainTransformerModule<B>,
    rope_cos: Tensor<B, 2>,
    rope_sin: Tensor<B, 2>,
    config: ModelConfig,
    data: Option<DataLoader>,
    step: u64,
    last_loss: f64,
    total_tokens: u64,
    start_params: Option<HashMap<String, Vec<f32>>>,
    start_shapes: Option<HashMap<String, Vec<usize>>>,
    error_buffer: ErrorBuffer,
}

// --- Response types ---

#[derive(Serialize, Deserialize)]
struct StatusResponse {
    initialized: bool,
    step: u64,
    last_loss: f64,
    total_tokens: u64,
    model_preset: String,
    gpu: bool,
}

#[derive(Serialize, Deserialize)]
struct StepResult { loss: f64, step: u64, tokens_processed: u64 }

#[derive(Serialize, Deserialize)]
struct CalibrationResult { secs_per_step: f64, recommended_h_mini: u64 }

#[derive(Serialize, Deserialize)]
struct ErrorResponse { error: String }

// --- Helpers ---

fn to_json_cstring<T: Serialize>(val: &T) -> *mut c_char {
    let json = serde_json::to_string(val).unwrap_or_else(|e| {
        format!(r#"{{"error":"serialization failed: {}"}}"#, e)
    });
    CString::new(json).unwrap_or_default().into_raw()
}

fn error_cstring(msg: &str) -> *mut c_char {
    to_json_cstring(&ErrorResponse { error: msg.to_string() })
}

/// Initialize model from config on a specific backend.
fn init_model<B: BurnBackend>(config: &ModelConfig, device: &B::Device) -> (DistrainTransformerModule<B>, Tensor<B, 2>, Tensor<B, 2>) {
    let (rope_cos, rope_sin) = precompute_rope_tables::<B>(
        config.head_dim(), config.max_seq_len, config.rope_theta, device,
    );
    let module = DistrainTransformerModule::<B>::new(config, device);
    (module, rope_cos, rope_sin)
}

fn new_state<B: BurnBackend>(config: ModelConfig, module: DistrainTransformerModule<B>, rope_cos: Tensor<B, 2>, rope_sin: Tensor<B, 2>, error_buffer: ErrorBuffer) -> StateInner<B> {
    StateInner {
        module, rope_cos, rope_sin, config,
        data: None, step: 0, last_loss: 0.0, total_tokens: 0,
        start_params: None, start_shapes: None, error_buffer,
    }
}

// --- Public FFI API ---

/// Initialize with a model preset. Probes GPU on first call; falls back to CPU if GPU fails.
/// Returns JSON with `gpu: true/false` indicating which backend is active.
#[no_mangle]
pub extern "C" fn distrain_init(preset_json: *const c_char) -> *mut c_char {
    let preset_str = unsafe {
        if preset_json.is_null() { return error_cstring("null preset"); }
        match CStr::from_ptr(preset_json).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return error_cstring(&format!("invalid UTF-8: {e}")),
        }
    };

    let preset = match preset_str.trim_matches('"') {
        "MicroTest" | "micro" | "micro-test" => ModelPreset::MicroTest,
        "Tiny" | "tiny" => ModelPreset::Tiny,
        "Small" | "small" => ModelPreset::Small,
        "Medium" | "medium" => ModelPreset::Medium,
        _ => return error_cstring(&format!("unknown preset: {preset_str}")),
    };
    let config = preset.config();

    let gpu = using_gpu();
    if gpu {
        let device = gpu_device();
        let (module, rope_cos, rope_sin) = init_model::<GpuBackend>(&config, &device);
        *GPU_STATE.lock().unwrap() = Some(new_state(config, module, rope_cos, rope_sin, ErrorBuffer::new()));
    } else {
        let device = cpu_device();
        let (module, rope_cos, rope_sin) = init_model::<CpuBackend>(&config, &device);
        *CPU_STATE.lock().unwrap() = Some(new_state(config, module, rope_cos, rope_sin, ErrorBuffer::new()));
    }

    to_json_cstring(&StatusResponse {
        initialized: true, step: 0, last_loss: 0.0, total_tokens: 0,
        model_preset: preset_str, gpu,
    })
}

/// Load training data from shard files.
#[no_mangle]
pub extern "C" fn distrain_load_shards(paths_json: *const c_char, batch_size: usize, seq_len: usize) -> *mut c_char {
    let json_str = unsafe {
        if paths_json.is_null() { return error_cstring("null paths"); }
        match CStr::from_ptr(paths_json).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return error_cstring(&format!("invalid UTF-8: {e}")),
        }
    };
    let paths: Vec<String> = match serde_json::from_str(&json_str) {
        Ok(p) => p,
        Err(e) => return error_cstring(&format!("invalid JSON: {e}")),
    };
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    let loader = match DataLoader::from_files(&path_bufs, batch_size, seq_len) {
        Ok(l) => l,
        Err(e) => return error_cstring(&e),
    };

    with_state!(|s| {
        s.data = Some(loader);
        to_json_cstring(&serde_json::json!({"ok": true, "shards": paths.len()}))
    })
}

/// Run one training step.
#[no_mangle]
pub extern "C" fn distrain_train_step(learning_rate: f64) -> *mut c_char {
    with_state!(|s| {
        let data = match s.data.as_mut() {
            Some(d) => d,
            None => return error_cstring("no data loaded"),
        };
        let batch_size = data.batch_size;
        let seq_len = data.seq_len;
        let tokens = data.next_batch();

        let device = s.rope_cos.device();
        let batch = Tensor::from_data(TensorData::new(tokens, [batch_size, seq_len]), &device);

        let loss = compute_lm_loss(&s.module, &s.rope_cos, &s.rope_sin, batch);
        let loss_val: f64 = loss.clone().into_scalar().elem();
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &s.module);

        let mut optim = AdamWConfig::new()
            .with_weight_decay(0.1)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init();

        let placeholder = DistrainTransformerModule::new(&s.config, &device);
        let module = std::mem::replace(&mut s.module, placeholder);
        s.module = optim.step(learning_rate, module, grads_params);

        s.step += 1;
        s.last_loss = loss_val;
        s.total_tokens += (batch_size * seq_len) as u64;

        to_json_cstring(&StepResult { loss: loss_val, step: s.step, tokens_processed: s.total_tokens })
    })
}

/// Get training status.
#[no_mangle]
pub extern "C" fn distrain_status() -> *mut c_char {
    let gpu = using_gpu();
    let (initialized, step, last_loss, total_tokens) = if gpu {
        match GPU_STATE.lock().unwrap().as_ref() {
            Some(s) => (true, s.step, s.last_loss, s.total_tokens),
            None => (false, 0, 0.0, 0),
        }
    } else {
        match CPU_STATE.lock().unwrap().as_ref() {
            Some(s) => (true, s.step, s.last_loss, s.total_tokens),
            None => (false, 0, 0.0, 0),
        }
    };
    to_json_cstring(&StatusResponse {
        initialized, step, last_loss, total_tokens,
        model_preset: if initialized { "active" } else { "none" }.to_string(),
        gpu,
    })
}

/// Calibrate device speed.
#[no_mangle]
pub extern "C" fn distrain_calibrate(target_interval_secs: f64) -> *mut c_char {
    let config = if using_gpu() {
        GPU_STATE.lock().ok().and_then(|g| g.as_ref().map(|s| s.config.clone()))
    } else {
        CPU_STATE.lock().ok().and_then(|g| g.as_ref().map(|s| s.config.clone()))
    }.unwrap_or_else(|| ModelPreset::MicroTest.config());

    let steps = 5u64;
    let seq_len = config.max_seq_len.min(32);
    let batch_size = 1usize;

    let elapsed = if using_gpu() {
        let device = gpu_device();
        calibrate_inner::<GpuBackend>(&config, &device, steps, batch_size, seq_len)
    } else {
        let device = cpu_device();
        calibrate_inner::<CpuBackend>(&config, &device, steps, batch_size, seq_len)
    };

    let secs_per_step = elapsed / steps as f64;
    let h_mini = ((target_interval_secs / secs_per_step) as u64).clamp(5, 1000);
    to_json_cstring(&CalibrationResult { secs_per_step, recommended_h_mini: h_mini })
}

fn calibrate_inner<B: burn::prelude::Backend + burn::tensor::backend::AutodiffBackend>(
    config: &ModelConfig, device: &B::Device, steps: u64, batch_size: usize, seq_len: usize,
) -> f64 {
    let (rope_cos, rope_sin) = precompute_rope_tables::<B>(
        config.head_dim(), config.max_seq_len, config.rope_theta, device,
    );
    let mut module = DistrainTransformerModule::<B>::new(config, device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let start = std::time::Instant::now();
    for step in 0..steps {
        let tokens: Vec<i64> = (0..(batch_size * seq_len))
            .map(|i| (splitmix64(step * 1000 + i as u64) % config.vocab_size as u64) as i64)
            .collect();
        let batch = Tensor::<B, 2, Int>::from_data(TensorData::new(tokens, [batch_size, seq_len]), device);
        let loss = compute_lm_loss(&module, &rope_cos, &rope_sin, batch);
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optim.step(3e-4, module, grads_params);
    }
    start.elapsed().as_secs_f64()
}

/// Load checkpoint from a file path (avoids holding entire checkpoint in caller memory).
#[no_mangle]
pub extern "C" fn distrain_load_checkpoint_file(path: *const c_char) -> *mut c_char {
    let path_str = unsafe {
        if path.is_null() { return error_cstring("null path"); }
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return error_cstring(&format!("invalid UTF-8: {e}")),
        }
    };
    let bytes = match std::fs::read(&path_str) {
        Ok(b) => b,
        Err(e) => return error_cstring(&format!("failed to read {path_str}: {e}")),
    };
    load_checkpoint_from_bytes(&bytes)
}

/// Load checkpoint from in-memory safetensors bytes.
#[no_mangle]
pub extern "C" fn distrain_load_checkpoint(data: *const u8, len: usize) -> *mut c_char {
    if data.is_null() || len == 0 {
        return error_cstring("null or empty checkpoint data");
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    load_checkpoint_from_bytes(bytes)
}

fn load_checkpoint_from_bytes(bytes: &[u8]) -> *mut c_char {

    let shapes = match load_safetensors_shapes_from_bytes(bytes) {
        Ok(s) => s,
        Err(e) => return error_cstring(&format!("failed to read checkpoint shapes: {e}")),
    };
    let config = match infer_model_config_from_shapes(&shapes) {
        Ok(c) => c,
        Err(e) => return error_cstring(&format!("failed to infer config: {e}")),
    };
    let state_dict = match load_safetensors_map_from_bytes(bytes) {
        Ok(s) => s,
        Err(e) => return error_cstring(&format!("failed to load checkpoint: {e}")),
    };

    let num_params = config.param_count();
    let max_seq_len = config.max_seq_len;
    let gpu = using_gpu();

    // Preserve error buffer
    let error_buffer = if gpu {
        GPU_STATE.lock().ok().and_then(|g| g.as_ref().map(|s| s.error_buffer.clone())).unwrap_or_default()
    } else {
        CPU_STATE.lock().ok().and_then(|g| g.as_ref().map(|s| s.error_buffer.clone())).unwrap_or_default()
    };

    if gpu {
        let device = gpu_device();
        let (module, rope_cos, rope_sin) = init_model::<GpuBackend>(&config, &device);
        let module = module.load_state_dict(&state_dict, &device);
        *GPU_STATE.lock().unwrap() = Some(new_state(config, module, rope_cos, rope_sin, error_buffer));
    } else {
        let device = cpu_device();
        let (module, rope_cos, rope_sin) = init_model::<CpuBackend>(&config, &device);
        let module = module.load_state_dict(&state_dict, &device);
        *CPU_STATE.lock().unwrap() = Some(new_state(config, module, rope_cos, rope_sin, error_buffer));
    }

    to_json_cstring(&serde_json::json!({"ok": true, "num_params": num_params, "max_seq_len": max_seq_len, "gpu": gpu}))
}

/// Snapshot params for delta computation.
#[no_mangle]
pub extern "C" fn distrain_snapshot_params() -> *mut c_char {
    with_state!(|s| {
        let params = s.module.extract_state_dict();
        let shapes = s.module.extract_shapes();
        let num_params: usize = params.values().map(|v| v.len()).sum();
        s.start_params = Some(params);
        s.start_shapes = Some(shapes);
        to_json_cstring(&serde_json::json!({"ok": true, "num_params": num_params}))
    })
}

/// Compute + compress delta, write to file.
#[no_mangle]
pub extern "C" fn distrain_compute_delta(output_path: *const c_char) -> *mut c_char {
    let path_str = unsafe {
        if output_path.is_null() { return error_cstring("null output path"); }
        match CStr::from_ptr(output_path).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return error_cstring(&format!("invalid UTF-8 path: {e}")),
        }
    };

    with_state!(|s| {
        let start_params = match &s.start_params {
            Some(p) => p,
            None => return error_cstring("no snapshot — call distrain_snapshot_params first"),
        };
        let shapes = match &s.start_shapes {
            Some(sh) => sh,
            None => return error_cstring("no snapshot shapes"),
        };

        let current_params = s.module.extract_state_dict();
        let delta = compute_outer_delta(start_params, &current_params);
        let compression_config = CompressionConfig::default();
        let (compressed, _stats) = match compress_delta(&delta, shapes, &compression_config, &mut s.error_buffer) {
            Ok(c) => c,
            Err(e) => return error_cstring(&format!("compression failed: {e}")),
        };
        let size_bytes = compressed.len();
        if let Err(e) = std::fs::write(&path_str, &compressed) {
            return error_cstring(&format!("failed to write delta: {e}"));
        }
        to_json_cstring(&serde_json::json!({"ok": true, "size_bytes": size_bytes}))
    })
}

/// Deterministic shard assignment (pure computation, no backend needed).
#[no_mangle]
pub extern "C" fn distrain_compute_shard_assignment(
    node_id: *const c_char, version: u64, total_shards: u32, shards_per_node: u32,
) -> *mut c_char {
    let node_id_str = unsafe {
        if node_id.is_null() { return error_cstring("null node_id"); }
        match CStr::from_ptr(node_id).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return error_cstring(&format!("invalid UTF-8: {e}")),
        }
    };
    let shards = compute_shard_assignment(&node_id_str, version, total_shards as usize, shards_per_node as usize);
    to_json_cstring(&shards)
}

// --- Remaining FFI functions (backend-independent) ---

#[no_mangle]
pub extern "C" fn distrain_shutdown() {
    *GPU_STATE.lock().unwrap() = None;
    *CPU_STATE.lock().unwrap() = None;
}

#[no_mangle]
pub unsafe extern "C" fn distrain_free_string(ptr: *mut c_char) {
    if !ptr.is_null() { let _ = CString::from_raw(ptr); }
}

#[no_mangle]
pub extern "C" fn distrain_cosine_lr(step: u64, warmup_steps: u64, total_steps: u64, max_lr: f64, min_lr: f64) -> f64 {
    cosine_lr(step as usize, warmup_steps as usize, total_steps as usize, max_lr, min_lr)
}

#[no_mangle]
pub extern "C" fn distrain_default_training_params() -> *mut c_char {
    to_json_cstring(&TrainingParams::default())
}

// --- Helpers ---

fn infer_model_config_from_shapes(shapes: &HashMap<String, Vec<usize>>) -> Result<ModelConfig, String> {
    let emb_shape = shapes.get("embedding.embedding.weight")
        .ok_or("Missing embedding.embedding.weight in checkpoint")?;
    let vocab_size = emb_shape[0];
    let hidden_dim = emb_shape[1];
    let num_layers = (0..).take_while(|i| shapes.contains_key(&format!("layers.{i}.attn_norm.weight"))).count();
    if num_layers == 0 { return Err("No layers found in checkpoint".to_string()); }
    let k_shape = shapes.get("layers.0.attention.k_proj.weight").ok_or("Missing k_proj.weight")?;
    let kv_dim = k_shape[0];
    let gate_shape = shapes.get("layers.0.ffn.gate_proj.weight").ok_or("Missing gate_proj.weight")?;
    let ffn_hidden_dim = gate_shape[0];
    let qkv_bias = shapes.contains_key("layers.0.attention.q_proj.bias");

    for preset in [ModelPreset::MicroTest, ModelPreset::Tiny, ModelPreset::Small, ModelPreset::Medium, ModelPreset::Large] {
        let cfg = preset.config();
        if cfg.hidden_dim == hidden_dim && cfg.num_layers == num_layers && cfg.vocab_size == vocab_size {
            return Ok(cfg);
        }
    }

    let head_dim = if hidden_dim % 128 == 0 && hidden_dim / 128 >= 2 { 128 } else { 64 };
    Ok(ModelConfig {
        hidden_dim, num_layers, num_heads: hidden_dim / head_dim, num_kv_heads: kv_dim / head_dim,
        vocab_size, max_seq_len: 4096, ffn_hidden_dim, rope_theta: 500_000.0, norm_eps: 1e-5,
        qkv_bias, attention_dropout: 0.0, tie_embeddings: true,
    })
}

fn splitmix64(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_init_and_status() {
        let preset = CString::new("MicroTest").unwrap();
        let result_ptr = distrain_init(preset.as_ptr());
        assert!(!result_ptr.is_null());
        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap().to_string() };
        unsafe { distrain_free_string(result_ptr) };

        let status: StatusResponse = serde_json::from_str(&result_str).unwrap();
        assert!(status.initialized);
        assert_eq!(status.step, 0);

        distrain_shutdown();
    }

    #[test]
    fn test_status_before_init() {
        distrain_shutdown();
        let result_ptr = distrain_status();
        let result_str = unsafe { CStr::from_ptr(result_ptr).to_str().unwrap().to_string() };
        unsafe { distrain_free_string(result_ptr) };

        let status: StatusResponse = serde_json::from_str(&result_str).unwrap();
        assert!(!status.initialized);
    }
}
