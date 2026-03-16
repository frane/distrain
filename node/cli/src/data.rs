//! Data shard reader — loads pre-tokenized binary shards for training.
//!
//! Shard format: flat little-endian uint16 token IDs, no header.
//! Manifest: `data/manifest.json` in R2 with shard metadata.
//!
//! Each node uses `compute_shard_assignment` to deterministically select
//! which shards to train on, ensuring different nodes see different data.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

use distrain_shared::storage::Storage;

/// Shard manifest loaded from R2.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub num_shards: usize,
    #[allow(dead_code)]
    pub vocab_size: Option<u32>,
    pub total_tokens: u64,
    pub shards: Vec<ShardEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShardEntry {
    pub filename: String,
    #[allow(dead_code)]
    pub num_tokens: u64,
    #[allow(dead_code)]
    pub size_bytes: u64,
}

/// Reads batches of token IDs from binary shard files.
pub struct DataLoader {
    shards: Vec<Vec<u16>>,
    seq_len: usize,
    batch_size: usize,
    /// Current position: (shard_index, offset_within_shard)
    shard_idx: usize,
    offset: usize,
    total_tokens: u64,
}

impl DataLoader {
    /// Download manifest from R2 and cache it locally. Returns (manifest, data_cache_dir).
    pub async fn load_manifest(
        storage: &Storage,
        cache_dir: &Path,
    ) -> Result<(Manifest, PathBuf)> {
        let data_cache = cache_dir.join("data");
        tokio::fs::create_dir_all(&data_cache).await?;

        let manifest_key = distrain_shared::paths::manifest_path();
        let manifest_path = data_cache.join("manifest.json");

        if !manifest_path.exists() {
            info!("Downloading data manifest...");
            storage
                .download_to_file(&manifest_key, &manifest_path)
                .await
                .context("Failed to download data manifest. Run prepare_data.py first.")?;
        }

        let manifest_text = tokio::fs::read_to_string(&manifest_path).await?;
        let manifest: Manifest =
            serde_json::from_str(&manifest_text).context("Failed to parse manifest.json")?;

        info!(
            "Data manifest: {} shards, {} total tokens",
            manifest.num_shards, manifest.total_tokens
        );

        Ok((manifest, data_cache))
    }

    /// Download assigned shards from R2 and build a DataLoader.
    ///
    /// Uses `compute_shard_assignment` to select shards deterministically
    /// based on node_id and checkpoint_version — different nodes get different data.
    pub async fn from_assignment(
        storage: &Storage,
        manifest: &Manifest,
        data_cache: &Path,
        shard_indices: &[usize],
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let mut shard_paths = Vec::new();

        for &idx in shard_indices {
            let entry = manifest
                .shards
                .get(idx)
                .with_context(|| format!("Shard index {idx} out of range (total: {})", manifest.shards.len()))?;
            let shard_key = format!("data/{}", entry.filename);
            let shard_path = data_cache.join(&entry.filename);

            if !shard_path.exists() {
                storage
                    .download_to_file(&shard_key, &shard_path)
                    .await
                    .with_context(|| format!("Failed to download shard {}", entry.filename))?;
            }
            shard_paths.push(shard_path);
        }

        Self::from_files(&shard_paths, seq_len, batch_size)
    }

    /// Download manifest and ALL shards from R2 to local cache, then load.
    /// Prefer `from_assignment` for distributed training (different data per node).
    pub async fn from_storage(
        storage: &Storage,
        cache_dir: &Path,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let (manifest, data_cache) = Self::load_manifest(storage, cache_dir).await?;
        let all_indices: Vec<usize> = (0..manifest.shards.len()).collect();
        Self::from_assignment(storage, &manifest, &data_cache, &all_indices, seq_len, batch_size).await
    }

    /// Load from local shard files.
    pub fn from_files(
        shard_paths: &[PathBuf],
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let mut shards = Vec::new();
        let mut total_tokens: u64 = 0;

        for path in shard_paths {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read shard {}", path.display()))?;

            if bytes.len() % 2 != 0 {
                anyhow::bail!("Shard {} has odd byte count", path.display());
            }
            let tokens: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();

            total_tokens += tokens.len() as u64;
            info!(
                "Loaded shard {} — {} tokens",
                path.file_name().unwrap_or_default().to_string_lossy(),
                tokens.len()
            );
            shards.push(tokens);
        }

        let tokens_per_batch = seq_len * batch_size;
        anyhow::ensure!(
            total_tokens as usize >= tokens_per_batch,
            "Not enough data: {total_tokens} tokens < {tokens_per_batch} needed per batch"
        );

        info!(
            "DataLoader ready: {} tokens across {} shards (batch_size={}, seq_len={})",
            total_tokens,
            shards.len(),
            batch_size,
            seq_len,
        );

        Ok(Self {
            shards,
            seq_len,
            batch_size,
            shard_idx: 0,
            offset: 0,
            total_tokens,
        })
    }

    /// Get the next batch of token IDs as a flat Vec<i64> with shape [batch_size, seq_len].
    ///
    /// Wraps around to the beginning when data is exhausted.
    pub fn next_batch(&mut self) -> Vec<i64> {
        self.next_batch_sized(self.batch_size)
    }

    /// Get the next batch with a specific batch size (for gradient accumulation micro-batches).
    pub fn next_batch_sized(&mut self, batch_size: usize) -> Vec<i64> {
        let tokens_needed = batch_size * self.seq_len;
        let mut result = Vec::with_capacity(tokens_needed);

        while result.len() < tokens_needed {
            let shard = &self.shards[self.shard_idx];
            let remaining_in_shard = shard.len() - self.offset;
            let need = tokens_needed - result.len();
            let take = need.min(remaining_in_shard);

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

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Seek to a deterministic position based on a seed (e.g., seq_num).
    /// Ensures different rounds on the same checkpoint use different data.
    pub fn seek_by_seed(&mut self, seed: u64) {
        if self.shards.is_empty() {
            return;
        }
        // Use seed to pick a starting shard and offset
        let shard_idx = (seed as usize) % self.shards.len();
        let shard_len = self.shards[shard_idx].len();
        let offset = if shard_len > 0 {
            // Align to seq_len boundaries for clean sequences
            let pos = ((seed.wrapping_mul(0x9E3779B97F4A7C15)) as usize) % shard_len;
            (pos / self.seq_len) * self.seq_len
        } else {
            0
        };
        self.shard_idx = shard_idx;
        self.offset = offset;
        info!("DataLoader seek: seed={seed} → shard={shard_idx}, offset={offset}");
    }

    /// Reset to the beginning of the data (for eval).
    pub fn reset(&mut self) {
        self.shard_idx = 0;
        self.offset = 0;
    }
}
