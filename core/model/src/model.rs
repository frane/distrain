//! DistrainTransformer — full decoder-only transformer in Burn.
//!
//! Faithful port of the Python model:
//! - RMSNorm (pre-norm)
//! - GQA with RoPE (θ=500k)
//! - SwiGLU FFN
//! - Tied embeddings with output scaling
//! - DeepSeek-style initialization

use std::collections::HashMap;

use burn::module::Param;
use burn::prelude::*;
use burn::nn;
use burn::tensor::{activation, TensorData};

use crate::config::ModelConfig;

/// Transpose flat row-major data from `[rows, cols]` to `[cols, rows]` layout.
///
/// Used to convert between Burn's Linear weight layout `[d_input, d_output]`
/// and PyTorch's `[d_output, d_input]`.
fn transpose_flat(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ── RMSNorm ────────────────────────────────────────────────────────────

/// RMSNorm — simpler and faster than LayerNorm.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    #[module(skip)]
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [batch, seq, dim]
        let variance = x.clone().powf_scalar(2.0).mean_dim(2); // [batch, seq, 1]
        let rms = (variance + self.eps).sqrt().recip();
        let normed = x * rms;
        // weight: [dim] → [1, 1, dim] for broadcast
        let w: Tensor<B, 3> = self.weight.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        normed * w
    }
}

// ── RoPE ───────────────────────────────────────────────────────────────

/// Precompute RoPE sin/cos tables.
/// Returns (cos, sin) each of shape [max_seq_len, head_dim/2].
pub fn precompute_rope_tables<B: Backend>(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half_dim = head_dim / 2;

    let positions: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let freqs = Tensor::<B, 1>::from_floats(positions.as_slice(), device);

    let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let t = Tensor::<B, 1>::from_floats(t.as_slice(), device);

    // outer product: angles[pos, dim] = pos * freq[dim]
    let t_2d: Tensor<B, 2> = t.unsqueeze_dim::<2>(1);
    let freqs_2d: Tensor<B, 2> = freqs.unsqueeze_dim::<2>(0);
    let angles = t_2d * freqs_2d;

    let cos = angles.clone().cos();
    let sin = angles.sin();

    (cos, sin)
}

/// Apply rotary embeddings using real-valued sin/cos.
/// x: [batch, seq, heads, head_dim]
/// cos, sin: [max_seq, head_dim/2] — sliced to seq_len
///
/// Cos/sin are expanded via raw data construction to avoid repeat_dim,
/// which has backward issues in Burn 0.16's autodiff engine.
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
    start_pos: usize,
) -> Tensor<B, 4> {
    let [batch, seq_len, heads, head_dim] = x.dims();
    let half = head_dim / 2;

    // cos/sin: [S, half] → [1, S, 1, half] for broadcasting over batch and heads
    let cos_slice = cos.clone().slice([start_pos..start_pos + seq_len, 0..half]);
    let sin_slice = sin.clone().slice([start_pos..start_pos + seq_len, 0..half]);
    let cos_4d = cos_slice.reshape([1, seq_len, 1, half]).expand([batch, seq_len, heads, half]);
    let sin_4d = sin_slice.reshape([1, seq_len, 1, half]).expand([batch, seq_len, heads, half]);

    // Split x into first half and second half
    let x1 = x.clone().slice([0..batch, 0..seq_len, 0..heads, 0..half]);
    let x2 = x.slice([0..batch, 0..seq_len, 0..heads, half..head_dim]);

    let out1 = x1.clone() * cos_4d.clone() - x2.clone() * sin_4d.clone();
    let out2 = x1 * sin_4d + x2 * cos_4d;

    Tensor::cat(vec![out1, out2], 3)
}

/// Expand KV heads for GQA by constructing the expanded tensor from raw data.
/// Avoids repeat_dim which has backward issues in Burn 0.16's autodiff.
/// Input: [B, S, kv_heads, D] → Output: [B, S, kv_heads * groups, D]
fn expand_kv_heads<B: Backend>(x: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
    if groups == 1 {
        return x;
    }
    let [batch, seq_len, kv_heads, dim] = x.dims();
    let num_heads = kv_heads * groups;

    // [B, S, kv, D] → [B, S, kv, 1, D] → [B, S, kv, groups, D] → [B, S, kv*groups, D]
    x.reshape([batch, seq_len, kv_heads, 1, dim])
        .expand([batch, seq_len, kv_heads, groups, dim])
        .reshape([batch, seq_len, num_heads, dim])
}

/// Build additive causal mask from raw data: 0.0 for allowed, -1e9 for masked.
/// Shape: [B, H, S, S]. No autodiff tracking on the mask tensor.
fn build_causal_mask<B: Backend>(
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let per_head = seq_len * seq_len;
    let total = batch * num_heads * per_head;
    let mut mask = vec![0.0f32; total];

    for b in 0..batch {
        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    let idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    mask[idx] = -1e4;
                }
            }
        }
    }

    Tensor::from_data(
        TensorData::new(mask, [batch, num_heads, seq_len, seq_len]),
        device,
    )
}

// ── Grouped Query Attention ────────────────────────────────────────────

/// GQA with RoPE. QKV bias from Qwen 2.5.
#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    o_proj: nn::Linear<B>,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    num_gqa_groups: usize,
}

impl<B: Backend> GroupedQueryAttention<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let kv_dim = config.num_kv_heads * config.head_dim();

        Self {
            q_proj: nn::LinearConfig::new(config.hidden_dim, config.num_heads * config.head_dim())
                .with_bias(config.qkv_bias)
                .init(device),
            k_proj: nn::LinearConfig::new(config.hidden_dim, kv_dim)
                .with_bias(config.qkv_bias)
                .init(device),
            v_proj: nn::LinearConfig::new(config.hidden_dim, kv_dim)
                .with_bias(config.qkv_bias)
                .init(device),
            o_proj: nn::LinearConfig::new(config.num_heads * config.head_dim(), config.hidden_dim)
                .with_bias(false)
                .init(device),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim(),
            num_gqa_groups: config.num_gqa_groups(),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 2>,
        sin: &Tensor<B, 2>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();

        // Project Q, K, V
        let q = self.q_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self.k_proj.forward(x.clone())
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = self.v_proj.forward(x)
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // Apply RoPE
        let q = apply_rope(q, cos, sin, start_pos);
        let k = apply_rope(k, cos, sin, start_pos);

        // Expand KV heads for GQA via raw data (avoids repeat_dim backward bug)
        let (k, v) = if self.num_kv_heads != self.num_heads {
            (expand_kv_heads(k, self.num_gqa_groups), expand_kv_heads(v, self.num_gqa_groups))
        } else {
            (k, v)
        };

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Scaled dot-product attention with causal mask
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Causal mask: construct from raw data to avoid repeat_dim backward issues.
        let additive_mask = build_causal_mask::<B>(
            batch, self.num_heads, seq_len, &scores.device(),
        );
        let scores = scores + additive_mask;

        let attn_weights = activation::softmax(scores, 3);
        let attn_out = attn_weights.matmul(v);

        // Reshape back: [batch, heads, seq, head_dim] → [batch, seq, hidden]
        let attn_out = attn_out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.o_proj.forward(attn_out)
    }
}

// ── SwiGLU FFN ─────────────────────────────────────────────────────────

/// SwiGLU(x) = SiLU(xW_gate) ⊙ xW_up, then W_down.
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            gate_proj: nn::LinearConfig::new(config.hidden_dim, config.ffn_hidden_dim)
                .with_bias(false)
                .init(device),
            up_proj: nn::LinearConfig::new(config.hidden_dim, config.ffn_hidden_dim)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(config.ffn_hidden_dim, config.hidden_dim)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ── Tied Embedding ─────────────────────────────────────────────────────

/// Tied input/output embedding with learnable output scaling (Qwen 2.5).
#[derive(Module, Debug)]
pub struct TiedEmbedding<B: Backend> {
    embedding: nn::Embedding<B>,
    output_scale: Param<Tensor<B, 1>>,
    #[module(skip)]
    tie: bool,
}

impl<B: Backend> TiedEmbedding<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            embedding: nn::EmbeddingConfig::new(config.vocab_size, config.hidden_dim).init(device),
            output_scale: Param::from_tensor(Tensor::ones([1], device)),
            tie: config.tie_embeddings,
        }
    }

    pub fn encode(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(token_ids)
    }

    pub fn decode(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.tie {
            let [_batch, _seq, _hdim] = hidden.dims();
            // Scale hidden states
            let scale: Tensor<B, 3> = self.output_scale.val()
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0);
            let scaled = hidden * scale;

            // weight: [vocab, hidden] → need [batch, vocab, hidden] for batched matmul
            let weight = self.embedding.weight.val(); // [vocab, hidden]
            let weight_t = weight.transpose(); // [hidden, vocab]
            // Expand weight to 3D: [1, hidden, vocab]
            let weight_3d: Tensor<B, 3> = weight_t.unsqueeze_dim::<3>(0);

            // [batch, seq, hidden] @ [1, hidden, vocab] → [batch, seq, vocab]
            scaled.matmul(weight_3d)
        } else {
            unimplemented!("Non-tied embeddings not yet supported in Burn port")
        }
    }
}

// ── Transformer Block ──────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attn_norm: RmsNorm<B>,
    attention: GroupedQueryAttention<B>,
    ffn_norm: RmsNorm<B>,
    ffn: SwiGluFfn<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            attn_norm: RmsNorm::new(config.hidden_dim, config.norm_eps, device),
            attention: GroupedQueryAttention::new(config, device),
            ffn_norm: RmsNorm::new(config.hidden_dim, config.norm_eps, device),
            ffn: SwiGluFfn::new(config, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: &Tensor<B, 2>,
        sin: &Tensor<B, 2>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let h = x.clone() + self.attention.forward(self.attn_norm.forward(x.clone()), cos, sin, start_pos);
        h.clone() + self.ffn.forward(self.ffn_norm.forward(h))
    }
}

// ── Full Model ─────────────────────────────────────────────────────────

/// The trainable module (derives Module for Burn's parameter system).
#[derive(Module, Debug)]
pub struct DistrainTransformerModule<B: Backend> {
    pub embedding: TiedEmbedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub final_norm: RmsNorm<B>,
}

impl<B: Backend> DistrainTransformerModule<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| TransformerBlock::new(config, device))
            .collect();

        Self {
            embedding: TiedEmbedding::new(config, device),
            layers,
            final_norm: RmsNorm::new(config.hidden_dim, config.norm_eps, device),
        }
    }

    /// Extract all parameters as flat f32 vectors (state dict).
    ///
    /// Key naming matches Python exactly:
    ///   embedding.embedding.weight, embedding.output_scale,
    ///   layers.{i}.attn_norm.weight, layers.{i}.attention.{q,k,v,o}_proj.{weight,bias},
    ///   layers.{i}.ffn_norm.weight, layers.{i}.ffn.{gate,up,down}_proj.weight,
    ///   final_norm.weight
    ///
    /// Linear weight matrices are transposed from Burn layout `[d_input, d_output]`
    /// to PyTorch layout `[d_output, d_input]` for cross-framework compatibility.
    pub fn extract_state_dict(&self) -> HashMap<String, Vec<f32>> {
        let mut state = HashMap::new();

        // Embedding (same layout in Burn and PyTorch — no transpose needed)
        let emb_w: Vec<f32> = self.embedding.embedding.weight.val().into_data().to_vec().unwrap();
        state.insert("embedding.embedding.weight".to_string(), emb_w);
        let out_scale: Vec<f32> = self.embedding.output_scale.val().into_data().to_vec().unwrap();
        state.insert("embedding.output_scale".to_string(), out_scale);

        // Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{i}");

            // Attention norms (1D — no transpose)
            let norm_w: Vec<f32> = layer.attn_norm.weight.val().into_data().to_vec().unwrap();
            state.insert(format!("{prefix}.attn_norm.weight"), norm_w);

            // Attention projections (2D Linear weights — transpose to PyTorch layout)
            for (name, proj) in [
                ("q_proj", &layer.attention.q_proj),
                ("k_proj", &layer.attention.k_proj),
                ("v_proj", &layer.attention.v_proj),
                ("o_proj", &layer.attention.o_proj),
            ] {
                let dims = proj.weight.val().dims();
                let w: Vec<f32> = proj.weight.val().into_data().to_vec().unwrap();
                let w = transpose_flat(&w, dims[0], dims[1]);
                state.insert(format!("{prefix}.attention.{name}.weight"), w);
                if let Some(ref bias) = proj.bias {
                    let b: Vec<f32> = bias.val().into_data().to_vec().unwrap();
                    state.insert(format!("{prefix}.attention.{name}.bias"), b);
                }
            }

            // FFN norm (1D — no transpose)
            let ffn_norm_w: Vec<f32> = layer.ffn_norm.weight.val().into_data().to_vec().unwrap();
            state.insert(format!("{prefix}.ffn_norm.weight"), ffn_norm_w);

            // FFN projections (2D Linear weights — transpose to PyTorch layout)
            for (name, proj) in [
                ("gate_proj", &layer.ffn.gate_proj),
                ("up_proj", &layer.ffn.up_proj),
                ("down_proj", &layer.ffn.down_proj),
            ] {
                let dims = proj.weight.val().dims();
                let w: Vec<f32> = proj.weight.val().into_data().to_vec().unwrap();
                let w = transpose_flat(&w, dims[0], dims[1]);
                state.insert(format!("{prefix}.ffn.{name}.weight"), w);
            }
        }

        // Final norm (1D — no transpose)
        let final_norm_w: Vec<f32> = self.final_norm.weight.val().into_data().to_vec().unwrap();
        state.insert("final_norm.weight".to_string(), final_norm_w);

        state
    }

    /// Async variant of `extract_state_dict` for WASM/WebGPU.
    ///
    /// On WASM, `into_data()` returns a Future because GPU buffer readback is async.
    /// Native builds use the sync `extract_state_dict()` instead.
    #[cfg(target_family = "wasm")]
    pub async fn extract_state_dict_async(&self) -> HashMap<String, Vec<f32>> {
        let mut state = HashMap::new();

        let emb_w: Vec<f32> = self.embedding.embedding.weight.val().into_data_async().await.expect("async readback").to_vec().unwrap();
        state.insert("embedding.embedding.weight".to_string(), emb_w);
        let out_scale: Vec<f32> = self.embedding.output_scale.val().into_data_async().await.expect("async readback").to_vec().unwrap();
        state.insert("embedding.output_scale".to_string(), out_scale);

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{i}");

            let norm_w: Vec<f32> = layer.attn_norm.weight.val().into_data_async().await.expect("async readback").to_vec().unwrap();
            state.insert(format!("{prefix}.attn_norm.weight"), norm_w);

            for (name, proj) in [
                ("q_proj", &layer.attention.q_proj),
                ("k_proj", &layer.attention.k_proj),
                ("v_proj", &layer.attention.v_proj),
                ("o_proj", &layer.attention.o_proj),
            ] {
                let dims = proj.weight.val().dims();
                let w: Vec<f32> = proj.weight.val().into_data_async().await.expect("async readback").to_vec().unwrap();
                let w = transpose_flat(&w, dims[0], dims[1]);
                state.insert(format!("{prefix}.attention.{name}.weight"), w);
                if let Some(ref bias) = proj.bias {
                    let b: Vec<f32> = bias.val().into_data_async().await.expect("async readback").to_vec().unwrap();
                    state.insert(format!("{prefix}.attention.{name}.bias"), b);
                }
            }

            let ffn_norm_w: Vec<f32> = layer.ffn_norm.weight.val().into_data_async().await.expect("async readback").to_vec().unwrap();
            state.insert(format!("{prefix}.ffn_norm.weight"), ffn_norm_w);

            for (name, proj) in [
                ("gate_proj", &layer.ffn.gate_proj),
                ("up_proj", &layer.ffn.up_proj),
                ("down_proj", &layer.ffn.down_proj),
            ] {
                let dims = proj.weight.val().dims();
                let w: Vec<f32> = proj.weight.val().into_data_async().await.expect("async readback").to_vec().unwrap();
                let w = transpose_flat(&w, dims[0], dims[1]);
                state.insert(format!("{prefix}.ffn.{name}.weight"), w);
            }
        }

        let final_norm_w: Vec<f32> = self.final_norm.weight.val().into_data_async().await.expect("async readback").to_vec().unwrap();
        state.insert("final_norm.weight".to_string(), final_norm_w);

        state
    }

    /// Extract parameter shapes matching the state dict keys.
    ///
    /// Shapes are in PyTorch convention: Linear weights are `[d_output, d_input]`.
    pub fn extract_shapes(&self) -> HashMap<String, Vec<usize>> {
        let mut shapes = HashMap::new();

        // Embedding — same layout in both frameworks
        shapes.insert(
            "embedding.embedding.weight".to_string(),
            self.embedding.embedding.weight.val().dims().to_vec(),
        );
        shapes.insert(
            "embedding.output_scale".to_string(),
            self.embedding.output_scale.val().dims().to_vec(),
        );

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{i}");
            shapes.insert(
                format!("{prefix}.attn_norm.weight"),
                layer.attn_norm.weight.val().dims().to_vec(),
            );

            // Linear weights: transpose shape from Burn [in, out] → PyTorch [out, in]
            for (name, proj) in [
                ("q_proj", &layer.attention.q_proj),
                ("k_proj", &layer.attention.k_proj),
                ("v_proj", &layer.attention.v_proj),
                ("o_proj", &layer.attention.o_proj),
            ] {
                let dims = proj.weight.val().dims();
                shapes.insert(
                    format!("{prefix}.attention.{name}.weight"),
                    vec![dims[1], dims[0]], // transposed
                );
                if let Some(ref bias) = proj.bias {
                    shapes.insert(
                        format!("{prefix}.attention.{name}.bias"),
                        bias.val().dims().to_vec(),
                    );
                }
            }

            shapes.insert(
                format!("{prefix}.ffn_norm.weight"),
                layer.ffn_norm.weight.val().dims().to_vec(),
            );

            // Linear weights: transpose shape
            for (name, proj) in [
                ("gate_proj", &layer.ffn.gate_proj),
                ("up_proj", &layer.ffn.up_proj),
                ("down_proj", &layer.ffn.down_proj),
            ] {
                let dims = proj.weight.val().dims();
                shapes.insert(
                    format!("{prefix}.ffn.{name}.weight"),
                    vec![dims[1], dims[0]], // transposed
                );
            }
        }

        shapes.insert(
            "final_norm.weight".to_string(),
            self.final_norm.weight.val().dims().to_vec(),
        );

        shapes
    }

    /// Load parameters from a state dict, returning a new module.
    ///
    /// The state dict keys must match the naming convention from `extract_state_dict()`.
    /// Linear weight data is expected in PyTorch layout `[d_output, d_input]` and is
    /// transposed to Burn layout `[d_input, d_output]`.
    pub fn load_state_dict(
        mut self,
        state: &HashMap<String, Vec<f32>>,
        device: &B::Device,
    ) -> Self {
        // Helper: create a 1D tensor from flat data
        fn make_1d<B: Backend>(data: &[f32], device: &B::Device) -> Tensor<B, 1> {
            Tensor::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
        }
        // Helper: create a 2D tensor from flat data
        fn make_2d<B: Backend>(data: &[f32], shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
            Tensor::from_data(TensorData::new(data.to_vec(), shape), device)
        }

        // Embedding (same layout in both frameworks — no transpose)
        if let Some(w) = state.get("embedding.embedding.weight") {
            let dims = self.embedding.embedding.weight.val().dims();
            self.embedding.embedding.weight =
                Param::from_tensor(make_2d::<B>(w, [dims[0], dims[1]], device));
        }
        if let Some(s) = state.get("embedding.output_scale") {
            self.embedding.output_scale = Param::from_tensor(make_1d::<B>(s, device));
        }

        // Layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{i}");

            if let Some(w) = state.get(&format!("{prefix}.attn_norm.weight")) {
                layer.attn_norm.weight = Param::from_tensor(make_1d::<B>(w, device));
            }

            // Attention projections — transpose from PyTorch [out, in] → Burn [in, out]
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let w_key = format!("{prefix}.attention.{name}.weight");
                if let Some(w) = state.get(&w_key) {
                    let burn_dims = match name {
                        "q_proj" => layer.attention.q_proj.weight.val().dims(),
                        "k_proj" => layer.attention.k_proj.weight.val().dims(),
                        "v_proj" => layer.attention.v_proj.weight.val().dims(),
                        "o_proj" => layer.attention.o_proj.weight.val().dims(),
                        _ => unreachable!(),
                    };
                    // Input is PyTorch layout [out, in]; transpose to Burn [in, out]
                    let pytorch_rows = burn_dims[1]; // d_output in PyTorch dim 0
                    let pytorch_cols = burn_dims[0]; // d_input in PyTorch dim 1
                    let transposed = transpose_flat(w, pytorch_rows, pytorch_cols);
                    let t = make_2d::<B>(&transposed, [burn_dims[0], burn_dims[1]], device);
                    match name {
                        "q_proj" => layer.attention.q_proj.weight = Param::from_tensor(t),
                        "k_proj" => layer.attention.k_proj.weight = Param::from_tensor(t),
                        "v_proj" => layer.attention.v_proj.weight = Param::from_tensor(t),
                        "o_proj" => layer.attention.o_proj.weight = Param::from_tensor(t),
                        _ => unreachable!(),
                    }
                }

                let b_key = format!("{prefix}.attention.{name}.bias");
                if let Some(b) = state.get(&b_key) {
                    let t = make_1d::<B>(b, device);
                    match name {
                        "q_proj" => layer.attention.q_proj.bias = Some(Param::from_tensor(t)),
                        "k_proj" => layer.attention.k_proj.bias = Some(Param::from_tensor(t)),
                        "v_proj" => layer.attention.v_proj.bias = Some(Param::from_tensor(t)),
                        "o_proj" => layer.attention.o_proj.bias = Some(Param::from_tensor(t)),
                        _ => unreachable!(),
                    }
                }
            }

            if let Some(w) = state.get(&format!("{prefix}.ffn_norm.weight")) {
                layer.ffn_norm.weight = Param::from_tensor(make_1d::<B>(w, device));
            }

            // FFN projections — transpose from PyTorch [out, in] → Burn [in, out]
            for name in ["gate_proj", "up_proj", "down_proj"] {
                let w_key = format!("{prefix}.ffn.{name}.weight");
                if let Some(w) = state.get(&w_key) {
                    let burn_dims = match name {
                        "gate_proj" => layer.ffn.gate_proj.weight.val().dims(),
                        "up_proj" => layer.ffn.up_proj.weight.val().dims(),
                        "down_proj" => layer.ffn.down_proj.weight.val().dims(),
                        _ => unreachable!(),
                    };
                    let pytorch_rows = burn_dims[1];
                    let pytorch_cols = burn_dims[0];
                    let transposed = transpose_flat(w, pytorch_rows, pytorch_cols);
                    let t = make_2d::<B>(&transposed, [burn_dims[0], burn_dims[1]], device);
                    match name {
                        "gate_proj" => layer.ffn.gate_proj.weight = Param::from_tensor(t),
                        "up_proj" => layer.ffn.up_proj.weight = Param::from_tensor(t),
                        "down_proj" => layer.ffn.down_proj.weight = Param::from_tensor(t),
                        _ => unreachable!(),
                    }
                }
            }
        }

        // Final norm
        if let Some(w) = state.get("final_norm.weight") {
            self.final_norm.weight = Param::from_tensor(make_1d::<B>(w, device));
        }

        self
    }
}

/// DistrainTransformer — decoder-only transformer, Option D best-of-breed.
///
/// Wraps the trainable module with non-trainable state (RoPE tables, config).
pub struct DistrainTransformer<B: Backend> {
    pub module: DistrainTransformerModule<B>,
    pub rope_cos: Tensor<B, 2>,
    pub rope_sin: Tensor<B, 2>,
    pub config: ModelConfig,
}

impl<B: Backend> DistrainTransformer<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        let (rope_cos, rope_sin) = precompute_rope_tables::<B>(
            config.head_dim(),
            config.max_seq_len,
            config.rope_theta,
            device,
        );

        Self {
            module: DistrainTransformerModule::new(config, device),
            rope_cos,
            rope_sin,
            config: config.clone(),
        }
    }

    /// Forward pass: token_ids [batch, seq] → logits [batch, seq, vocab].
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_at(token_ids, 0)
    }

    /// Forward with explicit start position (for KV cache / incremental decode).
    pub fn forward_at(&self, token_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3> {
        let mut h = self.module.embedding.encode(token_ids);

        for layer in &self.module.layers {
            h = layer.forward(h, &self.rope_cos, &self.rope_sin, start_pos);
        }

        h = self.module.final_norm.forward(h);
        self.module.embedding.decode(h)
    }

    /// Compute cross-entropy loss for language modeling.
    pub fn compute_loss(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [batch, seq_len] = token_ids.dims();

        // Input: all tokens except last
        let input = token_ids.clone().slice([0..batch, 0..seq_len - 1]);
        let logits = self.forward(input); // [batch, seq-1, vocab]

        // Target: all tokens except first
        let targets = token_ids.slice([0..batch, 1..seq_len]);

        // Reshape for cross-entropy
        let [b, s, v] = logits.dims();
        let logits_flat = logits.reshape([b * s, v]); // [N, vocab]
        let targets_flat = targets.reshape([b * s]); // [N]

        // Manual cross-entropy: -mean(log_softmax(logits)[targets])
        let log_probs = activation::log_softmax(logits_flat, 1); // [N, vocab]
        let target_indices: Tensor<B, 2, Int> = targets_flat.unsqueeze_dim::<2>(1); // [N, 1]
        let gathered = log_probs.gather(1, target_indices); // [N, 1]
        gathered.mean().neg() // scalar loss
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// Compute language modeling loss given model components separately.
///
/// This allows the training loop to own the module for optimizer steps
/// while keeping RoPE tables and config as separate references.
pub fn compute_lm_loss<B: Backend>(
    module: &DistrainTransformerModule<B>,
    rope_cos: &Tensor<B, 2>,
    rope_sin: &Tensor<B, 2>,
    token_ids: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch, seq_len] = token_ids.dims();

    // Input: all tokens except last
    let input = token_ids.clone().slice([0..batch, 0..seq_len - 1]);
    // Target: all tokens except first
    let targets = token_ids.slice([0..batch, 1..seq_len]);

    // Forward pass through model components
    let mut h = module.embedding.encode(input);
    for layer in &module.layers {
        h = layer.forward(h, rope_cos, rope_sin, 0);
    }
    h = module.final_norm.forward(h);
    let logits = module.embedding.decode(h);

    // Cross-entropy loss
    let [b, s, v] = logits.dims();
    let logits_flat = logits.reshape([b * s, v]);
    let targets_flat = targets.reshape([b * s]);
    let log_probs = activation::log_softmax(logits_flat, 1);
    let target_indices: Tensor<B, 2, Int> = targets_flat.unsqueeze_dim::<2>(1);
    let gathered = log_probs.gather(1, target_indices);
    gathered.mean().neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            vocab_size: 256,
            max_seq_len: 128,
            ffn_hidden_dim: 128,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
            qkv_bias: true,
            attention_dropout: 0.0,
            tie_embeddings: true,
        }
    }

    #[test]
    fn test_rmsnorm_output_shape() {
        let device = Default::default();
        let norm = RmsNorm::<TestBackend>::new(64, 1e-5, &device);
        let x = Tensor::random([2, 8, 64], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let out = norm.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_rope_tables_shape() {
        let device = Default::default();
        let (cos, sin) = precompute_rope_tables::<TestBackend>(64, 128, 500_000.0, &device);
        assert_eq!(cos.dims(), [128, 32]);
        assert_eq!(sin.dims(), [128, 32]);
    }

    #[test]
    fn test_swiglu_output_shape() {
        let device = Default::default();
        let config = tiny_config();
        let ffn = SwiGluFfn::<TestBackend>::new(&config, &device);
        let x = Tensor::random([2, 8, 64], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_gqa_output_shape() {
        let device = Default::default();
        let config = tiny_config();
        let attn = GroupedQueryAttention::<TestBackend>::new(&config, &device);
        let (cos, sin) = precompute_rope_tables::<TestBackend>(16, 128, 500_000.0, &device);
        let x = Tensor::random([2, 8, 64], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let out = attn.forward(x, &cos, &sin, 0);
        assert_eq!(out.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_transformer_forward() {
        let device = Default::default();
        let config = tiny_config();
        let model = DistrainTransformer::<TestBackend>::new(&config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4]], &device);
        let logits = model.forward(tokens);
        assert_eq!(logits.dims(), [1, 4, 256]);
    }

    #[test]
    fn test_transformer_loss() {
        let device = Default::default();
        let config = tiny_config();
        let model = DistrainTransformer::<TestBackend>::new(&config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_ints([[1, 2, 3, 4, 5]], &device);
        let loss = model.compute_loss(tokens);
        let loss_val: f32 = loss.into_scalar().elem();
        assert!(loss_val > 0.0, "Loss should be positive, got {loss_val}");
    }

    #[test]
    fn test_extract_state_dict_keys() {
        let device = Default::default();
        let config = tiny_config();
        let module = DistrainTransformerModule::<TestBackend>::new(&config, &device);
        let state = module.extract_state_dict();

        // Should have embedding keys
        assert!(state.contains_key("embedding.embedding.weight"));
        assert!(state.contains_key("embedding.output_scale"));
        assert!(state.contains_key("final_norm.weight"));

        // Should have keys for each layer
        for i in 0..config.num_layers {
            assert!(state.contains_key(&format!("layers.{i}.attn_norm.weight")));
            assert!(state.contains_key(&format!("layers.{i}.attention.q_proj.weight")));
            assert!(state.contains_key(&format!("layers.{i}.attention.k_proj.weight")));
            assert!(state.contains_key(&format!("layers.{i}.attention.v_proj.weight")));
            assert!(state.contains_key(&format!("layers.{i}.attention.o_proj.weight")));
            assert!(state.contains_key(&format!("layers.{i}.ffn_norm.weight")));
            assert!(state.contains_key(&format!("layers.{i}.ffn.gate_proj.weight")));
            assert!(state.contains_key(&format!("layers.{i}.ffn.up_proj.weight")));
            assert!(state.contains_key(&format!("layers.{i}.ffn.down_proj.weight")));
        }

        // Bias keys should exist when qkv_bias = true
        if config.qkv_bias {
            assert!(state.contains_key("layers.0.attention.q_proj.bias"));
            assert!(state.contains_key("layers.0.attention.k_proj.bias"));
            assert!(state.contains_key("layers.0.attention.v_proj.bias"));
        }
    }

    #[test]
    fn test_extract_load_roundtrip() {
        let device = Default::default();
        let config = tiny_config();
        let module = DistrainTransformerModule::<TestBackend>::new(&config, &device);

        // Extract state dict
        let state = module.extract_state_dict();

        // Create a new module and load the state dict
        let module2 = DistrainTransformerModule::<TestBackend>::new(&config, &device);
        let module2 = module2.load_state_dict(&state, &device);

        // Verify values match
        let state2 = module2.extract_state_dict();
        for (key, val) in &state {
            let val2 = &state2[key];
            assert_eq!(val.len(), val2.len(), "Length mismatch for {key}");
            for (i, (a, b)) in val.iter().zip(val2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "Mismatch at {key}[{i}]: {a} != {b}"
                );
            }
        }
    }

    #[test]
    fn test_extract_shapes() {
        let device = Default::default();
        let config = tiny_config();
        let module = DistrainTransformerModule::<TestBackend>::new(&config, &device);
        let shapes = module.extract_shapes();

        // Embedding weight should be [vocab_size, hidden_dim] (same in both frameworks)
        assert_eq!(shapes["embedding.embedding.weight"], vec![256, 64]);
        // Final norm weight should be [hidden_dim]
        assert_eq!(shapes["final_norm.weight"], vec![64]);
        // Q proj weight in PyTorch convention: [d_output, d_input] = [hidden_dim, hidden_dim]
        assert_eq!(shapes["layers.0.attention.q_proj.weight"], vec![64, 64]);
        // K proj weight in PyTorch convention: [kv_dim, hidden_dim] = [32, 64]
        assert_eq!(shapes["layers.0.attention.k_proj.weight"], vec![32, 64]);
        // Gate proj in PyTorch convention: [ffn_hidden, hidden_dim] = [128, 64]
        assert_eq!(shapes["layers.0.ffn.gate_proj.weight"], vec![128, 64]);
    }
}
