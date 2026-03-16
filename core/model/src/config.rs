//! Model configuration — all hyperparameters in one place.
//!
//! Mirrors Python `distrain.model.config.ModelConfig` exactly.

use serde::{Deserialize, Serialize};

/// Model hyperparameters. Matches Python ModelConfig field-for-field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub ffn_hidden_dim: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub qkv_bias: bool,
    pub attention_dropout: f64,
    pub tie_embeddings: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    pub fn num_gqa_groups(&self) -> usize {
        assert!(self.num_heads % self.num_kv_heads == 0);
        self.num_heads / self.num_kv_heads
    }

    /// Approximate parameter count (matches Python `param_count()`).
    pub fn param_count(&self) -> usize {
        let emb = self.vocab_size * self.hidden_dim;
        let q_proj = self.hidden_dim * self.hidden_dim;
        let kv_dim = self.head_dim() * self.num_kv_heads;
        let k_proj = self.hidden_dim * kv_dim;
        let v_proj = self.hidden_dim * kv_dim;
        let o_proj = self.hidden_dim * self.hidden_dim;
        let attn_bias = if self.qkv_bias {
            self.hidden_dim + kv_dim * 2
        } else {
            0
        };
        let attn = q_proj + k_proj + v_proj + o_proj + attn_bias;
        let ffn = 3 * self.hidden_dim * self.ffn_hidden_dim;
        let norms = 2 * self.hidden_dim;
        let block = attn + ffn + norms;
        let mut total = emb + self.num_layers * block + self.hidden_dim; // +hidden_dim for final_norm
        if !self.tie_embeddings {
            total += emb;
        }
        total
    }
}

/// Named presets matching Python CONFIGS dict.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelPreset {
    /// Micro test model (64 hidden, 2 layers) — for E2E testing only.
    #[serde(rename = "micro")]
    MicroTest,
    #[serde(rename = "125m")]
    Tiny,
    #[serde(rename = "1b")]
    Small,
    #[serde(rename = "7b")]
    Medium,
    #[serde(rename = "13b")]
    Large,
}

impl ModelPreset {
    pub fn config(self) -> ModelConfig {
        match self {
            Self::MicroTest => ModelConfig {
                hidden_dim: 64,
                num_layers: 2,
                num_heads: 4,
                num_kv_heads: 2,
                ffn_hidden_dim: 128,
                vocab_size: 256,
                max_seq_len: 64,
                rope_theta: 500_000.0,
                norm_eps: 1e-5,
                qkv_bias: true,
                attention_dropout: 0.0,
                tie_embeddings: true,
            },
            Self::Tiny => ModelConfig {
                hidden_dim: 768,
                num_layers: 12,
                num_heads: 12,
                num_kv_heads: 4,
                ffn_hidden_dim: 2048,
                vocab_size: 32768,
                max_seq_len: 4096,
                rope_theta: 500_000.0,
                norm_eps: 1e-5,
                qkv_bias: true,
                attention_dropout: 0.0,
                tie_embeddings: true,
            },
            Self::Small => ModelConfig {
                hidden_dim: 2048,
                num_layers: 24,
                num_heads: 16,
                num_kv_heads: 4,
                ffn_hidden_dim: 5504,
                vocab_size: 32768,
                max_seq_len: 4096,
                rope_theta: 500_000.0,
                norm_eps: 1e-5,
                qkv_bias: true,
                attention_dropout: 0.0,
                tie_embeddings: true,
            },
            Self::Medium => ModelConfig {
                hidden_dim: 4096,
                num_layers: 32,
                num_heads: 32,
                num_kv_heads: 8,
                ffn_hidden_dim: 11008,
                vocab_size: 32768,
                max_seq_len: 4096,
                rope_theta: 500_000.0,
                norm_eps: 1e-5,
                qkv_bias: true,
                attention_dropout: 0.0,
                tie_embeddings: true,
            },
            Self::Large => ModelConfig {
                hidden_dim: 5120,
                num_layers: 40,
                num_heads: 40,
                num_kv_heads: 8,
                ffn_hidden_dim: 13824,
                vocab_size: 32768,
                max_seq_len: 4096,
                rope_theta: 500_000.0,
                norm_eps: 1e-5,
                qkv_bias: true,
                attention_dropout: 0.0,
                tie_embeddings: true,
            },
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "micro" | "micro-test" => Some(Self::MicroTest),
            "125m" | "tiny" => Some(Self::Tiny),
            "1b" | "small" => Some(Self::Small),
            "7b" | "medium" => Some(Self::Medium),
            "13b" | "large" => Some(Self::Large),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_dim() {
        let cfg = ModelPreset::Medium.config();
        assert_eq!(cfg.head_dim(), 128);
    }

    #[test]
    fn test_gqa_groups() {
        let cfg = ModelPreset::Medium.config();
        assert_eq!(cfg.num_gqa_groups(), 4);
    }

    #[test]
    fn test_param_count_125m() {
        let cfg = ModelPreset::Tiny.config();
        let count = cfg.param_count();
        // Should be ~124M
        assert!(count > 100_000_000 && count < 150_000_000, "Got {count}");
    }

    #[test]
    fn test_all_presets() {
        for preset in [ModelPreset::Tiny, ModelPreset::Small, ModelPreset::Medium, ModelPreset::Large] {
            let cfg = preset.config();
            assert!(cfg.param_count() > 0);
            assert!(cfg.head_dim() > 0);
            assert!(cfg.num_gqa_groups() >= 1);
        }
    }
}
