//! Data shard reader — loads pre-tokenized binary shards for training.
//!
//! Shard format: flat little-endian uint16 token IDs, no header.
//! Manifest: `data/manifest.json` in R2 with shard metadata.
//!
//! Each node uses `compute_shard_assignment` to deterministically select
//! which shards to train on, ensuring different nodes see different data.
//!
//! Two loaders are provided:
//! - `DataLoader`: loads all assigned shards upfront (used for eval/baseline).
//! - `StreamingDataLoader`: downloads shards on demand, keeps only a few in memory.
//!   Training starts after the first 2 shards are ready instead of waiting for all 220.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{info, warn};

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

    /// Build a DataLoader from in-memory shard token data.
    ///
    /// Used by StreamingDataLoader to create a DataLoader that trainer functions can consume,
    /// without re-reading files from disk.
    pub fn from_tokens(
        shards: Vec<Vec<u16>>,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<Self> {
        let total_tokens: u64 = shards.iter().map(|s| s.len() as u64).sum();
        let tokens_per_batch = seq_len * batch_size;
        anyhow::ensure!(
            total_tokens as usize >= tokens_per_batch,
            "Not enough data: {total_tokens} tokens < {tokens_per_batch} needed per batch"
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

// ---------------------------------------------------------------------------
// StreamingDataLoader — downloads shards on demand, keeps only a few in memory
// ---------------------------------------------------------------------------

/// Streaming data loader: downloads shards on demand, keeps only a few in memory.
/// Training starts after first 2 shards are ready instead of waiting for all 220.
pub struct StreamingDataLoader {
    storage: Storage,
    /// Full list of assigned shard filenames (R2 keys without "data/" prefix).
    shard_names: Vec<String>,
    cache_dir: PathBuf,
    seq_len: usize,
    /// Currently loaded shard data: (index into shard_names, tokens).
    loaded: Vec<(usize, Vec<u16>)>,
    /// Index into `loaded` vec for the current read position.
    current_idx: usize,
    /// Token offset within the current loaded shard.
    offset: usize,
    /// Next shard_names index to download.
    next_to_download: usize,
    /// Maximum number of shards to keep in memory simultaneously.
    max_loaded: usize,
}

impl StreamingDataLoader {
    /// Create a new streaming data loader.
    ///
    /// Downloads the first `max_loaded` shards (or fewer if not that many assigned),
    /// then returns immediately. Remaining shards are downloaded on demand as training
    /// consumes data.
    pub async fn new(
        storage: Storage,
        shard_names: Vec<String>,
        cache_dir: PathBuf,
        seq_len: usize,
        max_loaded: usize,
    ) -> Result<Self> {
        let max_loaded = max_loaded.max(2); // always keep at least 2
        let initial_count = max_loaded.min(shard_names.len());

        let mut loaded = Vec::with_capacity(max_loaded);
        for i in 0..initial_count {
            let tokens = download_shard_cached(&storage, &shard_names[i], &cache_dir).await
                .with_context(|| format!("Failed to download initial shard {}", shard_names[i]))?;
            info!(
                "StreamingDataLoader: loaded shard {}/{} ({}) — {} tokens",
                i + 1,
                shard_names.len(),
                shard_names[i],
                tokens.len(),
            );
            loaded.push((i, tokens));
        }

        let total_loaded_tokens: u64 = loaded.iter().map(|(_, t)| t.len() as u64).sum();
        info!(
            "StreamingDataLoader ready: {} tokens across {}/{} shards (streaming remaining on demand)",
            total_loaded_tokens, initial_count, shard_names.len(),
        );

        Ok(Self {
            storage,
            shard_names,
            cache_dir,
            seq_len,
            loaded,
            current_idx: 0,
            offset: 0,
            next_to_download: initial_count,
            max_loaded,
        })
    }

    /// Download the next shard if we are running low on loaded data.
    ///
    /// Called between batches. If there are more shards to download and we have
    /// consumed past the first loaded shard, this evicts the oldest shard and
    /// downloads the next one.
    pub async fn ensure_next_shard(&mut self) -> Result<()> {
        // Nothing more to download
        if self.next_to_download >= self.shard_names.len() {
            return Ok(());
        }

        // Only download if we've moved past the first shard (i.e., current_idx > 0)
        // or if we have room. This keeps the pipeline fed.
        if self.loaded.len() >= self.max_loaded && self.current_idx == 0 {
            // All slots full and we haven't consumed the first one yet — no action needed.
            return Ok(());
        }

        // Evict consumed shards (everything before current_idx)
        if self.current_idx > 0 {
            let evicted_count = self.current_idx;
            self.loaded.drain(..evicted_count);
            self.current_idx = 0;
            // Fill the freed slots
            let slots_available = self.max_loaded.saturating_sub(self.loaded.len());
            for _ in 0..slots_available {
                if self.next_to_download >= self.shard_names.len() {
                    break;
                }
                let idx = self.next_to_download;
                let name = &self.shard_names[idx];
                match download_shard_cached(&self.storage, name, &self.cache_dir).await {
                    Ok(tokens) => {
                        info!(
                            "StreamingDataLoader: streamed shard {}/{} ({}) — {} tokens",
                            idx + 1,
                            self.shard_names.len(),
                            name,
                            tokens.len(),
                        );
                        self.loaded.push((idx, tokens));
                        self.next_to_download += 1;
                    }
                    Err(e) => {
                        warn!("Failed to download shard {} (skipping): {e:#}", name);
                        self.next_to_download += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the next batch of token IDs with the given batch size.
    ///
    /// Returns a flat `Vec<i64>` with shape `[batch_size, seq_len]`.
    /// When the current shard is exhausted, advances to the next loaded shard.
    /// Wraps around within loaded shards if all loaded shards are consumed.
    pub fn next_batch_sized(&mut self, batch_size: usize) -> Vec<i64> {
        let tokens_needed = batch_size * self.seq_len;
        let mut result = Vec::with_capacity(tokens_needed);

        if self.loaded.is_empty() {
            // Should not happen if constructed properly, but be safe.
            return result;
        }

        while result.len() < tokens_needed {
            let (_, ref shard) = self.loaded[self.current_idx];
            let remaining_in_shard = shard.len().saturating_sub(self.offset);
            let need = tokens_needed - result.len();
            let take = need.min(remaining_in_shard);

            for &tok in &shard[self.offset..self.offset + take] {
                result.push(tok as i64);
            }
            self.offset += take;

            if self.offset >= shard.len() {
                // Move to next loaded shard, wrapping around if necessary
                self.current_idx = (self.current_idx + 1) % self.loaded.len();
                self.offset = 0;
            }
        }

        result
    }

    /// Total number of tokens currently loaded in memory.
    pub fn total_tokens_available(&self) -> u64 {
        self.loaded.iter().map(|(_, t)| t.len() as u64).sum()
    }

    /// Number of shards currently loaded in memory.
    pub fn shards_loaded(&self) -> usize {
        self.loaded.len()
    }

    /// Total number of assigned shards (loaded + not yet downloaded).
    pub fn total_shards(&self) -> usize {
        self.shard_names.len()
    }

    /// Current shard index within the loaded shards (for resume state).
    pub fn current_shard_index(&self) -> usize {
        if self.loaded.is_empty() {
            return 0;
        }
        self.loaded.get(self.current_idx).map(|(idx, _)| *idx).unwrap_or(0)
    }

    /// Current token offset within the current shard (for resume state).
    pub fn current_token_offset(&self) -> usize {
        self.offset
    }

    /// Extract loaded shard data as a map from shard filename to tokens.
    /// Used to pass pre-loaded data to a new loader on checkpoint change.
    pub fn take_loaded_shards(&mut self) -> HashMap<String, Vec<u16>> {
        let mut map = HashMap::new();
        for (idx, tokens) in self.loaded.drain(..) {
            if let Some(name) = self.shard_names.get(idx) {
                map.insert(name.clone(), tokens);
            }
        }
        map
    }

    /// Create a new loader reusing pre-loaded shard data where assignments overlap.
    /// Only downloads shards that aren't already in memory.
    pub async fn new_with_cache(
        storage: Storage,
        shard_names: Vec<String>,
        cache_dir: PathBuf,
        seq_len: usize,
        max_loaded: usize,
        mut preloaded: HashMap<String, Vec<u16>>,
    ) -> Result<Self> {
        let max_loaded = max_loaded.max(2);
        let initial_count = max_loaded.min(shard_names.len());
        let mut loaded = Vec::with_capacity(max_loaded);
        let mut reused = 0usize;

        for i in 0..initial_count {
            let name = &shard_names[i];
            if let Some(tokens) = preloaded.remove(name) {
                loaded.push((i, tokens));
                reused += 1;
            } else {
                let tokens = download_shard_cached(&storage, name, &cache_dir).await
                    .with_context(|| format!("Failed to download shard {name}"))?;
                loaded.push((i, tokens));
            }
        }

        let total_loaded_tokens: u64 = loaded.iter().map(|(_, t)| t.len() as u64).sum();
        info!(
            "StreamingDataLoader ready: {} tokens across {}/{} shards ({} reused from previous checkpoint)",
            total_loaded_tokens, initial_count, shard_names.len(), reused,
        );

        Ok(Self {
            storage,
            shard_names,
            cache_dir,
            seq_len,
            loaded,
            current_idx: 0,
            offset: 0,
            next_to_download: initial_count,
            max_loaded,
        })
    }

    /// Seek to a deterministic position based on a seed (e.g., seq_num).
    /// Ensures different rounds on the same checkpoint use different data.
    pub fn seek_by_seed(&mut self, seed: u64) {
        if self.loaded.is_empty() {
            return;
        }
        // Pick a starting position within loaded shards
        let shard_idx = (seed as usize) % self.loaded.len();
        let shard_len = self.loaded[shard_idx].1.len();
        let offset = if shard_len > 0 {
            let pos = ((seed.wrapping_mul(0x9E3779B97F4A7C15)) as usize) % shard_len;
            (pos / self.seq_len) * self.seq_len
        } else {
            0
        };
        self.current_idx = shard_idx;
        self.offset = offset;
        info!("StreamingDataLoader seek: seed={seed} → loaded_idx={shard_idx}, offset={offset}");
    }

    /// Create a DataLoader snapshot from the currently loaded shards.
    ///
    /// This clones the loaded shard data into a DataLoader that the trainer functions
    /// can consume (they expect `&mut DataLoader`). The streaming loader keeps its own
    /// copy for continued streaming in future rounds.
    pub fn to_data_loader(&self, batch_size: usize) -> Result<DataLoader> {
        let shards: Vec<Vec<u16>> = self.loaded.iter().map(|(_, t)| t.clone()).collect();
        DataLoader::from_tokens(shards, self.seq_len, batch_size)
    }
}

/// Download a shard to the cache directory if not already present, then load it.
async fn download_shard_cached(
    storage: &Storage,
    shard_filename: &str,
    cache_dir: &Path,
) -> Result<Vec<u16>> {
    let shard_path = cache_dir.join(shard_filename);

    if !shard_path.exists() {
        let shard_key = format!("data/{}", shard_filename);
        storage
            .download_to_file(&shard_key, &shard_path)
            .await
            .with_context(|| format!("Failed to download shard {shard_filename}"))?;
    }

    let bytes = tokio::fs::read(&shard_path)
        .await
        .with_context(|| format!("Failed to read shard {}", shard_path.display()))?;

    if bytes.len() % 2 != 0 {
        anyhow::bail!("Shard {} has odd byte count", shard_path.display());
    }

    let tokens: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    Ok(tokens)
}
