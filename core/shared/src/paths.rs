//! All S3/R2 path definitions. Never hardcode paths elsewhere.

pub const DEFAULT_BUCKET: &str = "distrain-training";

pub fn checkpoint_path(version: u64) -> String {
    format!("checkpoints/v{version}/model.safetensors")
}

pub fn checkpoint_metadata_path(version: u64) -> String {
    format!("checkpoints/v{version}/metadata.json")
}

pub fn optimizer_state_path(version: u64) -> String {
    format!("optimizer_state/v{version}/velocity.safetensors")
}

pub fn delta_path(version: u64, node_id: &str, seq_num: u64) -> String {
    format!("deltas/v{version}/{node_id}_{seq_num}.delta.zst")
}

pub fn accumulator_path() -> String {
    "accumulator/current.json".to_string()
}

pub fn data_shard_path(shard_id: u32) -> String {
    format!("data/shard_{shard_id:04}.bin")
}

pub fn manifest_path() -> String {
    "data/manifest.json".to_string()
}

pub fn run_config_path() -> String {
    "config/run_config.json".to_string()
}

pub fn node_registry_path() -> String {
    "state/node_registry.json".to_string()
}

pub fn outer_lr_state_path() -> String {
    "state/outer_lr.json".to_string()
}

pub fn stats_history_path() -> String {
    "stats/training_history.jsonl".to_string()
}

pub fn coordinator_state_path() -> String {
    "state/coordinator.json".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_path() {
        assert_eq!(checkpoint_path(5), "checkpoints/v5/model.safetensors");
    }

    #[test]
    fn test_delta_path() {
        assert_eq!(
            delta_path(3, "node_abc", 42),
            "deltas/v3/node_abc_42.delta.zst"
        );
    }

    #[test]
    fn test_shard_path_padding() {
        assert_eq!(data_shard_path(7), "data/shard_0007.bin");
        assert_eq!(data_shard_path(1234), "data/shard_1234.bin");
    }
}
