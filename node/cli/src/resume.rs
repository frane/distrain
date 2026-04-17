//! Node resume: save and restore training state across restarts.
//!
//! On every completed round, the node saves:
//! - Last checkpoint version used
//! - Data loader position (shard index, offset)
//! - Sequence number
//! - Error buffer (binary file, separate from state TOML)
//!
//! On restart, the node loads this state and resumes from where it left off.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Lightweight state saved as TOML after each completed round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    pub last_checkpoint_version: u64,
    pub seq_num: u64,
    /// Shard index within the assigned shards.
    pub shard_index: usize,
    /// Token offset within the current shard.
    pub shard_offset: usize,
    /// Node ID (persistent across restarts).
    pub node_id: String,
    /// Timestamp of last save (ISO 8601).
    pub saved_at: String,
}

impl NodeState {
    pub fn state_path(cache_dir: &Path) -> PathBuf {
        cache_dir.join("state.toml")
    }

    pub fn save(&self, cache_dir: &Path) -> Result<()> {
        let path = Self::state_path(cache_dir);
        let toml = toml::to_string_pretty(self).context("Failed to serialize node state")?;
        std::fs::write(&path, toml).context("Failed to write state.toml")?;
        Ok(())
    }

    pub fn load(cache_dir: &Path) -> Result<Option<NodeState>> {
        let path = Self::state_path(cache_dir);
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path).context("Failed to read state.toml")?;
        let state: NodeState = toml::from_str(&text).context("Failed to parse state.toml")?;
        Ok(Some(state))
    }
}

/// Save error buffer to a binary file.
///
/// Format: for each tensor, write [name_len: u32][name: bytes][count: u32][values: f32...].
/// This is much more compact than TOML/JSON for large buffers.
pub fn save_error_buffer(
    buffer: &distrain_model::compression::ErrorBuffer,
    cache_dir: &Path,
) -> Result<()> {
    let path = cache_dir.join("error_buffer.bin");
    let mut data = Vec::new();

    // Magic bytes + version
    data.extend_from_slice(b"DREB"); // Distrain Error Buffer
    data.extend_from_slice(&1u32.to_le_bytes()); // format version

    let num_tensors = buffer.buffer.len() as u32;
    data.extend_from_slice(&num_tensors.to_le_bytes());

    for (name, values) in &buffer.buffer {
        let name_bytes = name.as_bytes();
        data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(name_bytes);
        data.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    std::fs::write(&path, &data).context("Failed to write error buffer")?;
    let size_mb = data.len() as f64 / (1024.0 * 1024.0);
    info!("Saved error buffer: {:.1} MB ({num_tensors} tensors)", size_mb);
    Ok(())
}

/// Load error buffer from a binary file.
pub fn load_error_buffer(
    cache_dir: &Path,
) -> Result<Option<distrain_model::compression::ErrorBuffer>> {
    let path = cache_dir.join("error_buffer.bin");
    if !path.exists() {
        return Ok(None);
    }

    let data = std::fs::read(&path).context("Failed to read error buffer")?;
    if data.len() < 12 {
        warn!("Error buffer file too small, ignoring");
        return Ok(None);
    }

    // Check magic
    if &data[0..4] != b"DREB" {
        warn!("Error buffer file has wrong magic, ignoring");
        return Ok(None);
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 {
        warn!("Error buffer format version {version} not supported, ignoring");
        return Ok(None);
    }

    let num_tensors = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let mut offset = 12;
    let mut buffer = HashMap::new();

    for _ in 0..num_tensors {
        if offset + 4 > data.len() {
            warn!("Error buffer truncated at tensor name length");
            return Ok(None);
        }
        let name_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if offset + name_len > data.len() {
            warn!("Error buffer truncated at tensor name");
            return Ok(None);
        }
        let name = String::from_utf8_lossy(&data[offset..offset + name_len]).to_string();
        offset += name_len;

        if offset + 4 > data.len() {
            warn!("Error buffer truncated at value count");
            return Ok(None);
        }
        let count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let values_bytes = count * 4;
        if offset + values_bytes > data.len() {
            warn!("Error buffer truncated at values for {name}");
            return Ok(None);
        }
        let values: Vec<f32> = (0..count)
            .map(|i| {
                let start = offset + i * 4;
                f32::from_le_bytes(data[start..start + 4].try_into().unwrap())
            })
            .collect();
        offset += values_bytes;

        buffer.insert(name, values);
    }

    let size_mb = data.len() as f64 / (1024.0 * 1024.0);
    info!(
        "Loaded error buffer: {:.1} MB ({} tensors)",
        size_mb,
        buffer.len()
    );

    Ok(Some(distrain_model::compression::ErrorBuffer { buffer }))
}

/// Save importance tracker to a binary file (same format as error buffer).
pub fn save_importance_tracker(
    tracker: &distrain_model::importance::ImportanceTracker,
    cache_dir: &Path,
) -> Result<()> {
    let path = cache_dir.join("importance_tracker.bin");
    let mut data = Vec::new();

    // Magic bytes + version
    data.extend_from_slice(b"DRIT"); // Distrain Importance Tracker
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&tracker.alpha.to_le_bytes());

    let num_tensors = tracker.running_movement.len() as u32;
    data.extend_from_slice(&num_tensors.to_le_bytes());

    for (name, values) in &tracker.running_movement {
        let name_bytes = name.as_bytes();
        data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(name_bytes);
        data.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }

    std::fs::write(&path, &data).context("Failed to write importance tracker")?;
    let size_mb = data.len() as f64 / (1024.0 * 1024.0);
    info!("Saved importance tracker: {:.1} MB ({num_tensors} tensors)", size_mb);
    Ok(())
}

/// Load importance tracker from a binary file.
pub fn load_importance_tracker(
    cache_dir: &Path,
) -> Result<Option<distrain_model::importance::ImportanceTracker>> {
    let path = cache_dir.join("importance_tracker.bin");
    if !path.exists() {
        return Ok(None);
    }

    let data = std::fs::read(&path).context("Failed to read importance tracker")?;
    if data.len() < 16 {
        warn!("Importance tracker file too small, ignoring");
        return Ok(None);
    }

    if &data[0..4] != b"DRIT" {
        warn!("Importance tracker file has wrong magic, ignoring");
        return Ok(None);
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 {
        warn!("Importance tracker format version {version} not supported");
        return Ok(None);
    }

    let alpha = f32::from_le_bytes(data[8..12].try_into().unwrap());
    let num_tensors = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let mut offset = 16;
    let mut running_movement = HashMap::new();

    for _ in 0..num_tensors {
        if offset + 4 > data.len() { return Ok(None); }
        let name_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + name_len > data.len() { return Ok(None); }
        let name = String::from_utf8_lossy(&data[offset..offset + name_len]).to_string();
        offset += name_len;
        if offset + 4 > data.len() { return Ok(None); }
        let count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let values_bytes = count * 4;
        if offset + values_bytes > data.len() { return Ok(None); }
        let values: Vec<f32> = (0..count)
            .map(|i| {
                let start = offset + i * 4;
                f32::from_le_bytes(data[start..start + 4].try_into().unwrap())
            })
            .collect();
        offset += values_bytes;
        running_movement.insert(name, values);
    }

    info!("Loaded importance tracker: {} tensors, alpha={alpha}", running_movement.len());
    Ok(Some(distrain_model::importance::ImportanceTracker {
        running_movement,
        alpha,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let state = NodeState {
            last_checkpoint_version: 42,
            seq_num: 10,
            shard_index: 3,
            shard_offset: 1024,
            node_id: "node_abc123".to_string(),
            saved_at: "2026-04-17T00:00:00Z".to_string(),
        };
        state.save(dir.path()).unwrap();
        let loaded = NodeState::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.last_checkpoint_version, 42);
        assert_eq!(loaded.seq_num, 10);
        assert_eq!(loaded.shard_index, 3);
        assert_eq!(loaded.shard_offset, 1024);
        assert_eq!(loaded.node_id, "node_abc123");
    }

    #[test]
    fn test_error_buffer_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut buffer = distrain_model::compression::ErrorBuffer::new();
        buffer
            .buffer
            .insert("weight".to_string(), vec![0.1, -0.2, 0.3, 0.0, -0.5]);
        buffer
            .buffer
            .insert("bias".to_string(), vec![1.0, -1.0]);

        save_error_buffer(&buffer, dir.path()).unwrap();
        let loaded = load_error_buffer(dir.path()).unwrap().unwrap();

        assert_eq!(loaded.buffer.len(), 2);
        assert_eq!(loaded.buffer["weight"], vec![0.1, -0.2, 0.3, 0.0, -0.5]);
        assert_eq!(loaded.buffer["bias"], vec![1.0, -1.0]);
    }

    #[test]
    fn test_load_missing_state() {
        let dir = tempfile::tempdir().unwrap();
        assert!(NodeState::load(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_load_missing_error_buffer() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_error_buffer(dir.path()).unwrap().is_none());
    }

    #[test]
    fn test_importance_tracker_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut tracker = distrain_model::importance::ImportanceTracker::new(0.85);
        let mut delta = HashMap::new();
        delta.insert("weight".to_string(), vec![1.0, 2.0, 3.0]);
        tracker.update(&delta);

        save_importance_tracker(&tracker, dir.path()).unwrap();
        let loaded = load_importance_tracker(dir.path()).unwrap().unwrap();

        assert_eq!(loaded.alpha, 0.85);
        assert_eq!(loaded.running_movement["weight"].len(), 3);
        // After one update with alpha=0.85: m = 0.85*0 + 0.15*|d| = 0.15*d
        assert!((loaded.running_movement["weight"][0] - 0.15).abs() < 1e-5);
    }
}
