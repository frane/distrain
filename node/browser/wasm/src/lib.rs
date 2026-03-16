#![recursion_limit = "256"]
//! Distrain training engine compiled to WebAssembly.
//!
//! Uses WebGPU (burn-wgpu) when available, falls back to CPU (burn-ndarray).
//! JavaScript detects `navigator.gpu` and passes `use_gpu` flag to `wasm_init`.
//!
//! Build:
//!   wasm-pack build node/browser/wasm --target web --out-dir ../web/pkg

use std::collections::HashMap;
use std::sync::Mutex;

use burn::tensor::backend::AutodiffBackend;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use distrain_model::compression::{
    CompressionConfig, ErrorBuffer, build_sparse_delta, sparse_delta_to_json,
};
use distrain_model::config::ModelPreset;
use distrain_model::model::{compute_lm_loss, precompute_rope_tables, DistrainTransformerModule};
use distrain_model::training::compute_outer_delta;
use distrain_model::{GpuBackend, GpuDevice, CpuBackend, CpuDevice};

static STATE: Mutex<Option<WasmState>> = Mutex::new(None);

/// Backend-agnostic training state via enum dispatch.
/// All core training logic (model, compression, checkpointing) comes from core/model/.
/// This crate only provides the wasm-bindgen FFI boundary + backend selection.
enum WasmState {
    Gpu(TrainingState<GpuBackend>),
    Cpu(TrainingState<CpuBackend>),
}

struct TrainingState<B: AutodiffBackend> {
    module: DistrainTransformerModule<B>,
    rope_cos: Tensor<B, 2>,
    rope_sin: Tensor<B, 2>,
    config: distrain_model::config::ModelConfig,
    step: u64,
    last_loss: f64,
    total_tokens: u64,
    start_params: Option<HashMap<String, Vec<f32>>>,
    start_shapes: Option<HashMap<String, Vec<usize>>>,
    error_buffer: ErrorBuffer,
}

#[derive(Serialize, Deserialize)]
struct StatusResponse {
    initialized: bool,
    step: u64,
    last_loss: f64,
    total_tokens: u64,
    model_preset: String,
    backend: String,
    max_seq_len: usize,
}

#[derive(Serialize, Deserialize)]
struct StepResult {
    loss: f64,
    step: u64,
    tokens_processed: u64,
}

/// Macro to dispatch training operations to the correct backend.
/// Avoids duplicating every function body.
macro_rules! with_state {
    ($guard:expr, |$state:ident| $body:expr) => {
        match $guard.as_mut() {
            Some(WasmState::Gpu($state)) => $body,
            Some(WasmState::Cpu($state)) => $body,
            None => r#"{"error":"not initialized"}"#.to_string(),
        }
    };
}

macro_rules! with_state_ref {
    ($guard:expr, |$state:ident| $body:expr) => {
        match $guard.as_ref() {
            Some(WasmState::Gpu($state)) => $body,
            Some(WasmState::Cpu($state)) => $body,
            None => r#"{"error":"not initialized"}"#.to_string(),
        }
    };
}

fn init_state<B: AutodiffBackend>(
    preset: ModelPreset,
    device: &B::Device,
) -> TrainingState<B> {
    let config = preset.config();
    let (rope_cos, rope_sin) = precompute_rope_tables::<B>(
        config.head_dim(),
        config.max_seq_len,
        config.rope_theta,
        device,
    );
    let module = DistrainTransformerModule::<B>::new(&config, device);
    TrainingState {
        module,
        rope_cos,
        rope_sin,
        config,
        step: 0,
        last_loss: 0.0,
        total_tokens: 0,
        start_params: None,
        start_shapes: None,
        error_buffer: ErrorBuffer::new(),
    }
}

fn load_checkpoint_impl<B: AutodiffBackend>(
    state: &mut TrainingState<B>,
    data: &[u8],
    device: &B::Device,
) -> String {
    let state_dict = match distrain_model::checkpoint::load_safetensors_map_from_bytes(data) {
        Ok(sd) => sd,
        Err(e) => return format!(r#"{{"error":"failed to parse checkpoint: {}"}}"#, e),
    };
    let num_params: usize = state_dict.values().map(|v| v.len()).sum();
    let dummy = DistrainTransformerModule::<B>::new(&state.config, device);
    let module = std::mem::replace(&mut state.module, dummy);
    state.module = module.load_state_dict(&state_dict, device);
    format!(r#"{{"ok":true,"num_params":{num_params}}}"#)
}

fn train_step_impl<B: AutodiffBackend>(
    state: &mut TrainingState<B>,
    learning_rate: f64,
    token_data: &[u16],
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> String {
    if token_data.len() != batch_size * seq_len {
        return format!(
            r#"{{"error":"token_data length {} != batch_size*seq_len {}"}}"#,
            token_data.len(),
            batch_size * seq_len
        );
    }

    let tokens: Vec<i64> = token_data.iter().map(|&t| t as i64).collect();
    let batch = Tensor::<B, 2, Int>::from_data(
        TensorData::new(tokens, [batch_size, seq_len]),
        device,
    );

    let loss = compute_lm_loss(&state.module, &state.rope_cos, &state.rope_sin, batch);
    let loss_val: f64 = burn::tensor::ElementConversion::elem(loss.clone().into_scalar());

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &state.module);

    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<B, DistrainTransformerModule<B>>();

    let module = std::mem::replace(
        &mut state.module,
        DistrainTransformerModule::<B>::new(&state.config, device),
    );
    state.module = optim.step(learning_rate, module, grads_params);

    state.step += 1;
    state.last_loss = loss_val;
    state.total_tokens += (batch_size * seq_len) as u64;

    serde_json::to_string(&StepResult {
        loss: loss_val,
        step: state.step,
        tokens_processed: state.total_tokens,
    })
    .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e))
}

fn snapshot_impl<B: AutodiffBackend>(state: &mut TrainingState<B>) -> String {
    let params = state.module.extract_state_dict();
    let shapes = state.module.extract_shapes();
    let num_params: usize = params.values().map(|v| v.len()).sum();
    state.start_params = Some(params);
    state.start_shapes = Some(shapes);
    format!(r#"{{"ok":true,"num_params":{num_params}}}"#)
}

fn compute_delta_impl<B: AutodiffBackend>(state: &mut TrainingState<B>) -> String {
    let start_params = match &state.start_params {
        Some(p) => p,
        None => {
            return r#"{"error":"no snapshot taken — call wasm_snapshot_params() first"}"#
                .to_string()
        }
    };
    let start_shapes = match &state.start_shapes {
        Some(s) => s,
        None => return r#"{"error":"no shapes snapshot"}"#.to_string(),
    };

    let current_params = state.module.extract_state_dict();
    let delta = compute_outer_delta(start_params, &current_params);

    let config = CompressionConfig::default();
    let sparse = build_sparse_delta(&delta, start_shapes, &config, &mut state.error_buffer);

    match sparse_delta_to_json(&sparse) {
        Ok(json_bytes) => match String::from_utf8(json_bytes) {
            Ok(s) => s,
            Err(e) => format!(r#"{{"error":"utf8 error: {}"}}"#, e),
        },
        Err(e) => format!(r#"{{"error":"serialization failed: {}"}}"#, e),
    }
}

/// GPU training step — async because into_scalar() returns a Future on WASM/WebGPU.
async fn train_step_gpu(
    state: &mut TrainingState<GpuBackend>,
    learning_rate: f64,
    token_data: &[u16],
    batch_size: usize,
    seq_len: usize,
    device: &GpuDevice,
) -> String {
    if token_data.len() != batch_size * seq_len {
        return format!(
            r#"{{"error":"token_data length {} != batch_size*seq_len {}"}}"#,
            token_data.len(),
            batch_size * seq_len
        );
    }

    let tokens: Vec<i64> = token_data.iter().map(|&t| t as i64).collect();
    let batch = Tensor::<GpuBackend, 2, Int>::from_data(
        TensorData::new(tokens, [batch_size, seq_len]),
        device,
    );

    let loss = compute_lm_loss(&state.module, &state.rope_cos, &state.rope_sin, batch);
    // Read loss from GPU: get raw bytes via into_data_async, interpret as f32
    let loss_data = loss.clone().inner().into_data_async().await.expect("loss readback");
    let loss_val: f64 = match loss_data.to_vec::<f32>() {
        Ok(v) if !v.is_empty() => v[0] as f64,
        _ => -1.0,
    };

    let grads = loss.backward();
    let grads_params = GradientsParams::from_grads(grads, &state.module);

    // No GradientClippingConfig::Norm on WebGPU/WASM — it uses into_scalar() internally
    // which panics on async-only backends. Gradient clipping happens via weight_decay instead.
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.1)
        .init::<GpuBackend, DistrainTransformerModule<GpuBackend>>();

    let module = std::mem::replace(
        &mut state.module,
        DistrainTransformerModule::<GpuBackend>::new(&state.config, device),
    );
    state.module = optim.step(learning_rate, module, grads_params);

    state.step += 1;
    state.last_loss = loss_val;
    state.total_tokens += (batch_size * seq_len) as u64;

    serde_json::to_string(&StepResult {
        loss: loss_val,
        step: state.step,
        tokens_processed: state.total_tokens,
    })
    .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e))
}

// ── wasm-bindgen API ─────────────────────────────────────────────────

/// Initialize the training engine (async for WebGPU init on WASM).
/// `use_gpu`: true = WebGPU (fast), false = CPU/ndarray (slow fallback).
/// JavaScript should check `navigator.gpu` before deciding.
#[wasm_bindgen]
pub async fn wasm_init(preset: &str, use_gpu: bool) -> String {
    console_error_panic_hook::set_once();

    let preset = match preset {
        "MicroTest" | "micro" | "micro-test" => ModelPreset::MicroTest,
        "Tiny" | "tiny" => ModelPreset::Tiny,
        "Small" | "small" => ModelPreset::Small,
        "Medium" | "medium" => ModelPreset::Medium,
        _ => return format!(r#"{{"error":"unknown preset: {}"}}"#, preset),
    };

    let backend_name;
    let wasm_state = if use_gpu {
        // WebGPU requires async initialization on WASM
        use burn_wgpu::{init_setup_async, RuntimeOptions, graphics::WebGpu};
        let device: GpuDevice = Default::default();
        init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;
        backend_name = "WebGPU";
        WasmState::Gpu(init_state::<GpuBackend>(preset, &device))
    } else {
        let device: CpuDevice = Default::default();
        backend_name = "CPU (ndarray)";
        WasmState::Cpu(init_state::<CpuBackend>(preset, &device))
    };

    let max_seq_len = preset.config().max_seq_len;

    let mut guard = STATE.lock().unwrap();
    *guard = Some(wasm_state);

    serde_json::to_string(&StatusResponse {
        initialized: true,
        step: 0,
        last_loss: 0.0,
        total_tokens: 0,
        model_preset: preset.to_string(),
        backend: backend_name.to_string(),
        max_seq_len,
    })
    .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e))
}

/// Load checkpoint weights from safetensors bytes.
#[wasm_bindgen]
pub fn wasm_load_checkpoint(data: &[u8]) -> String {
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(e) => return format!(r#"{{"error":"lock failed: {}"}}"#, e),
    };

    match guard.as_mut() {
        Some(WasmState::Gpu(state)) => {
            let device: GpuDevice = Default::default();
            load_checkpoint_impl(state, data, &device)
        }
        Some(WasmState::Cpu(state)) => {
            let device: CpuDevice = Default::default();
            load_checkpoint_impl(state, data, &device)
        }
        None => r#"{"error":"not initialized — call wasm_init first"}"#.to_string(),
    }
}

/// Run one training step with real token data (async for WebGPU readback).
#[wasm_bindgen]
pub async fn wasm_train_step(
    learning_rate: f64,
    token_data: &[u16],
    batch_size: usize,
    seq_len: usize,
) -> String {
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(e) => return format!(r#"{{"error":"lock failed: {}"}}"#, e),
    };

    match guard.as_mut() {
        Some(WasmState::Gpu(state)) => {
            let device: GpuDevice = Default::default();
            train_step_gpu(state, learning_rate, token_data, batch_size, seq_len, &device).await
        }
        Some(WasmState::Cpu(state)) => {
            let device: CpuDevice = Default::default();
            train_step_impl(state, learning_rate, token_data, batch_size, seq_len, &device)
        }
        None => r#"{"error":"not initialized"}"#.to_string(),
    }
}

/// Snapshot current model parameters for delta computation (async for WebGPU).
#[wasm_bindgen]
pub async fn wasm_snapshot_params() -> String {
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(e) => return format!(r#"{{"error":"lock failed: {}"}}"#, e),
    };

    match guard.as_mut() {
        Some(WasmState::Gpu(state)) => {
            let params = state.module.extract_state_dict_async().await;
            let shapes = state.module.extract_shapes();
            let num_params: usize = params.values().map(|v| v.len()).sum();
            state.start_params = Some(params);
            state.start_shapes = Some(shapes);
            format!(r#"{{"ok":true,"num_params":{num_params}}}"#)
        }
        Some(WasmState::Cpu(state)) => snapshot_impl(state),
        None => r#"{"error":"not initialized"}"#.to_string(),
    }
}

/// Compute and return the sparse delta as JSON (async for WebGPU).
#[wasm_bindgen]
pub async fn wasm_compute_delta_json() -> String {
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(e) => return format!(r#"{{"error":"lock failed: {}"}}"#, e),
    };

    match guard.as_mut() {
        Some(WasmState::Gpu(state)) => {
            let start_params = match &state.start_params {
                Some(p) => p,
                None => return r#"{"error":"no snapshot"}"#.to_string(),
            };
            let start_shapes = match &state.start_shapes {
                Some(s) => s,
                None => return r#"{"error":"no shapes"}"#.to_string(),
            };
            let current_params = state.module.extract_state_dict_async().await;
            let delta = compute_outer_delta(start_params, &current_params);
            let config = CompressionConfig::default();
            let sparse = build_sparse_delta(&delta, start_shapes, &config, &mut state.error_buffer);
            match sparse_delta_to_json(&sparse) {
                Ok(json_bytes) => String::from_utf8(json_bytes).unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e)),
                Err(e) => format!(r#"{{"error":"{}"}}"#, e),
            }
        }
        Some(WasmState::Cpu(state)) => compute_delta_impl(state),
        None => r#"{"error":"not initialized"}"#.to_string(),
    }
}

/// Get current training status as JSON.
#[wasm_bindgen]
pub fn wasm_status() -> String {
    let guard = match STATE.lock() {
        Ok(g) => g,
        Err(e) => return format!(r#"{{"error":"lock failed: {}"}}"#, e),
    };

    with_state_ref!(guard, |state| {
        serde_json::to_string(&StatusResponse {
            initialized: true,
            step: state.step,
            last_loss: state.last_loss,
            total_tokens: state.total_tokens,
            model_preset: "active".to_string(),
            backend: "active".to_string(),
            max_seq_len: state.config.max_seq_len,
        })
        .unwrap_or_default()
    })
}

/// Shut down the training engine.
#[wasm_bindgen]
pub fn wasm_shutdown() {
    if let Ok(mut guard) = STATE.lock() {
        *guard = None;
    }
}

// ModelPreset doesn't implement Display
trait PresetName {
    fn to_string(&self) -> String;
}

impl PresetName for ModelPreset {
    fn to_string(&self) -> String {
        match self {
            ModelPreset::MicroTest => "MicroTest".to_string(),
            ModelPreset::Tiny => "Tiny".to_string(),
            ModelPreset::Small => "Small".to_string(),
            ModelPreset::Medium => "Medium".to_string(),
            ModelPreset::Large => "Large".to_string(),
        }
    }
}
