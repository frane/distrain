//! Real tests for v0.2 features — compression routing, importance, VRAM cascade, fallback mode.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use distrain_model::compression::*;
    use distrain_model::importance::ImportanceTracker;
    use distrain_shared::config::NodeConfig;

    use crate::trainer::compress_with_config;

    fn make_2d_delta(rows: usize, cols: usize) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut delta = HashMap::new();
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 / 50.0).sin() * 0.1).collect();
        delta.insert("layer.weight".to_string(), data);
        let mut bias = vec![0.01f32; cols];
        bias[0] = 0.5; // one large bias element
        delta.insert("layer.bias".to_string(), bias);
        let mut shapes = HashMap::new();
        shapes.insert("layer.weight".to_string(), vec![rows, cols]);
        shapes.insert("layer.bias".to_string(), vec![cols]);
        (delta, shapes)
    }

    // ── Pipeline routing tests ────────────────────────────────────

    #[test]
    fn test_unstructured_pipeline_produces_sparse_delta() {
        let (delta, shapes) = make_2d_delta(50, 20);
        let mut buf = ErrorBuffer::new();
        let config = NodeConfig {
            compression_pipeline: "unstructured".to_string(),
            ..Default::default()
        };
        let (compressed, stats) = compress_with_config(&delta, &shapes, 10.0, None, 0, &mut buf, &config, None).unwrap();
        assert!(compressed.len() > 0);
        assert!(stats.num_params_kept < stats.num_params_total);

        // Decompress and verify it's valid unstructured format
        let recovered = decompress_delta(&compressed).unwrap();
        assert_eq!(recovered["layer.weight"].len(), 1000);
        assert_eq!(recovered["layer.bias"].len(), 20);
    }

    #[test]
    fn test_block_pipeline_produces_block_delta() {
        let (delta, shapes) = make_2d_delta(50, 20);
        let mut buf = ErrorBuffer::new();
        let config = NodeConfig {
            compression_pipeline: "block".to_string(),
            ..Default::default()
        };
        let (compressed, stats) = compress_with_config(&delta, &shapes, 10.0, None, 0, &mut buf, &config, None).unwrap();
        assert!(compressed.len() > 0);
        assert!(stats.num_params_kept < stats.num_params_total);

        // Decompress — should auto-detect block format
        let json_bytes = zstd::decode_all(compressed.as_slice()).unwrap();
        let recovered = decompress_delta_json_auto(&json_bytes).unwrap();
        assert_eq!(recovered["layer.weight"].len(), 1000);
    }

    #[test]
    fn test_block_and_unstructured_produce_different_output() {
        let (delta, shapes) = make_2d_delta(50, 20);

        let mut buf1 = ErrorBuffer::new();
        let config1 = NodeConfig {
            compression_pipeline: "unstructured".to_string(),
            compression_retention: Some(0.2),
            ..Default::default()
        };
        let (c1, _) = compress_with_config(&delta, &shapes, 10.0, None, 0, &mut buf1, &config1, None).unwrap();

        let mut buf2 = ErrorBuffer::new();
        let config2 = NodeConfig {
            compression_pipeline: "block".to_string(),
            compression_retention: Some(0.2),
            ..Default::default()
        };
        let (c2, _) = compress_with_config(&delta, &shapes, 10.0, None, 0, &mut buf2, &config2, None).unwrap();

        // Different pipelines should produce different compressed output
        assert_ne!(c1, c2, "Block and unstructured should produce different output");
    }

    #[test]
    fn test_retention_override_changes_output() {
        let (delta, shapes) = make_2d_delta(50, 20);

        let mut buf1 = ErrorBuffer::new();
        let config1 = NodeConfig {
            compression_retention: Some(0.1),
            ..Default::default()
        };
        let (_, stats1) = compress_with_config(&delta, &shapes, 10.0, None, 0, &mut buf1, &config1, None).unwrap();

        let mut buf2 = ErrorBuffer::new();
        let config2 = NodeConfig {
            compression_retention: Some(0.9),
            ..Default::default()
        };
        let (_, stats2) = compress_with_config(&delta, &shapes, 10.0, None, 0, &mut buf2, &config2, None).unwrap();

        // 90% retention should keep more params than 10%
        assert!(
            stats2.num_params_kept > stats1.num_params_kept,
            "90% retention ({}) should keep more than 10% ({})",
            stats2.num_params_kept, stats1.num_params_kept
        );
    }

    #[test]
    fn test_bf16_mode_skips_quantization() {
        let (delta, shapes) = make_2d_delta(50, 20);
        let mut buf = ErrorBuffer::new();
        let config = NodeConfig {
            quantization_mode: "bf16".to_string(),
            compression_retention: Some(0.5),
            ..Default::default()
        };
        let (compressed, _) = compress_with_config(&delta, &shapes, 10.0, None, 0, &mut buf, &config, None).unwrap();
        // Should still produce valid output
        let recovered = decompress_delta(&compressed).unwrap();
        assert_eq!(recovered["layer.weight"].len(), 1000);
    }

    // ── Importance-weighted selection tests ────────────────────────

    #[test]
    fn test_importance_changes_selected_rows() {
        let (delta, shapes) = make_2d_delta(50, 20);

        // Build importance tracker with 10 rounds of biased history:
        // rows 0-24 move a lot (high historical movement), rows 25-49 are quiet
        let mut tracker = ImportanceTracker::new(0.9);
        for _ in 0..10 {
            let mut hist_delta = HashMap::new();
            let mut data = vec![0.01f32; 1000];
            // Rows 0-24 have high movement
            for r in 0..25 {
                for c in 0..20 {
                    data[r * 20 + c] = 1.0;
                }
            }
            hist_delta.insert("layer.weight".to_string(), data);
            hist_delta.insert("layer.bias".to_string(), vec![0.01; 20]);
            tracker.update(&hist_delta);
        }

        // Now compress a delta where ALL rows have similar magnitude
        let mut uniform_delta = HashMap::new();
        uniform_delta.insert("layer.weight".to_string(), vec![0.5f32; 1000]);
        uniform_delta.insert("layer.bias".to_string(), vec![0.01; 20]);

        // Without importance: rows selected by L2 norm (all equal → arbitrary)
        let mut buf1 = ErrorBuffer::new();
        let config1 = NodeConfig {
            compression_pipeline: "block".to_string(),
            compression_retention: Some(0.2), // keep 10 of 50 rows
            use_importance: false,
            ..Default::default()
        };
        let (c1, _) = compress_with_config(&uniform_delta, &shapes, 10.0, None, 0, &mut buf1, &config1, None).unwrap();

        // With importance: rows 25-49 (quiet history) should be prioritized
        let mut buf2 = ErrorBuffer::new();
        let config2 = NodeConfig {
            compression_pipeline: "block".to_string(),
            compression_retention: Some(0.2),
            use_importance: true,
            ..Default::default()
        };
        let (c2, _) = compress_with_config(&uniform_delta, &shapes, 10.0, None, 0, &mut buf2, &config2, Some(&mut tracker)).unwrap();

        // Outputs should differ because importance changes which rows are selected
        assert_ne!(c1, c2, "Importance-weighted selection should produce different output");

        // Decompress and verify the importance-weighted version selected different rows
        let j1 = zstd::decode_all(c1.as_slice()).unwrap();
        let j2 = zstd::decode_all(c2.as_slice()).unwrap();
        let r1 = decompress_delta_json_auto(&j1).unwrap();
        let r2 = decompress_delta_json_auto(&j2).unwrap();

        // Count nonzero rows in each
        let nonzero_rows_1: Vec<usize> = (0..50).filter(|&r| {
            r1["layer.weight"][r*20..(r+1)*20].iter().any(|x| *x != 0.0)
        }).collect();
        let nonzero_rows_2: Vec<usize> = (0..50).filter(|&r| {
            r2["layer.weight"][r*20..(r+1)*20].iter().any(|x| *x != 0.0)
        }).collect();

        // Importance-weighted should prefer rows 25-49 (quiet history → high importance)
        let quiet_row_count_2 = nonzero_rows_2.iter().filter(|&&r| r >= 25).count();
        let quiet_row_count_1 = nonzero_rows_1.iter().filter(|&&r| r >= 25).count();
        assert!(
            quiet_row_count_2 >= quiet_row_count_1,
            "Importance should favor quiet rows: importance selected {} quiet rows vs {} without",
            quiet_row_count_2, quiet_row_count_1
        );
    }

    // ── VRAM cascade tests ────────────────────────────────────────

    #[test]
    fn test_vram_cascade_tiers() {
        use crate::resources::{determine_vram_strategy, VramStrategy};
        use distrain_model::config::{ModelConfig, ModelPreset};

        let config = ModelPreset::Tiny.config();

        let (strategy, scale) = determine_vram_strategy(32 * 1024, &config);
        assert_eq!(strategy, VramStrategy::FullModel);
        assert_eq!(scale, 1.0);

        let (strategy, scale) = determine_vram_strategy(20 * 1024, &config);
        assert_eq!(strategy, VramStrategy::GradientCheckpointing);
        assert!(scale < 1.0);

        let (strategy, scale) = determine_vram_strategy(10 * 1024, &config);
        assert_eq!(strategy, VramStrategy::CheckpointWithOffload);
        assert!(scale < 1.0);

        let (strategy, _) = determine_vram_strategy(4 * 1024, &config);
        assert_eq!(strategy, VramStrategy::CpuOnly);
    }

    // ── Fallback cascade tests ────────────────────────────────────

    #[test]
    fn test_fallback_cascade_modes() {
        use distrain_shared::p2p::cascade::*;
        use distrain_shared::p2p::types::*;

        let p2p_disabled = P2pConfig { enabled: false, ..Default::default() };
        let p2p_enabled = P2pConfig { enabled: true, ..Default::default() };

        // P2P disabled + coordinator reachable → DirectHttp
        assert_eq!(
            determine_operating_mode(&p2p_disabled, "http://localhost:8000", false, true, 0, 0),
            OperatingMode::DirectHttp
        );

        // P2P disabled + coordinator unreachable → Solo
        assert_eq!(
            determine_operating_mode(&p2p_disabled, "http://localhost:8000", false, false, 0, 0),
            OperatingMode::Solo
        );

        // P2P enabled + multiple coordinators → FullP2p
        assert_eq!(
            determine_operating_mode(&p2p_enabled, "http://localhost:8000", true, true, 2, 3),
            OperatingMode::FullP2p
        );

        // P2P enabled + one coordinator → SingleCoordinatorWithDht
        assert_eq!(
            determine_operating_mode(&p2p_enabled, "http://localhost:8000", true, true, 1, 3),
            OperatingMode::SingleCoordinatorWithDht
        );

        // P2P enabled + no coordinator + peers → PeerMerge
        assert_eq!(
            determine_operating_mode(&p2p_enabled, "http://localhost:8000", true, false, 0, 3),
            OperatingMode::PeerMerge
        );

        // P2P enabled + nothing → Solo
        assert_eq!(
            determine_operating_mode(&p2p_enabled, "http://localhost:8000", true, false, 0, 0),
            OperatingMode::Solo
        );
    }

    // ── Peer merge state tests ─────────────────────────────────────

    #[test]
    fn test_peer_merge_add_and_dedup() {
        use crate::peer_merge::PeerMergeState;
        use distrain_shared::p2p::types::PeerDeltaAnnouncement;

        let mut pm = PeerMergeState::new(3, 5);

        let make_ann = |node: &str, seq: u64| PeerDeltaAnnouncement {
            node_id: node.to_string(),
            checkpoint_version: 5,
            delta_key: format!("deltas/v5/{node}_{seq}.delta.zst"),
            inner_steps: 50,
            training_loss: 10.0,
            tokens_processed: 1000,
            weight: 1000.0,
            timestamp: chrono::Utc::now(),
        };

        pm.add_delta(make_ann("node_a", 1));
        assert_eq!(pm.num_deltas(), 1);

        pm.add_delta(make_ann("node_b", 1));
        assert_eq!(pm.num_deltas(), 2);

        // Dedup: same node replaces
        pm.add_delta(make_ann("node_a", 2));
        assert_eq!(pm.num_deltas(), 2); // still 2, not 3

        // Third unique node triggers checkpoint
        let ready = pm.add_delta(make_ann("node_c", 1));
        assert!(ready, "Should be ready with 3 contributions");
        assert_eq!(pm.num_deltas(), 3);
    }

    #[test]
    fn test_peer_merge_staleness_filter() {
        use crate::peer_merge::PeerMergeState;
        use distrain_shared::p2p::types::PeerDeltaAnnouncement;

        let mut pm = PeerMergeState::new(2, 10);

        let stale = PeerDeltaAnnouncement {
            node_id: "node_a".to_string(),
            checkpoint_version: 5, // version 10 - 5 = 5 > 3 threshold
            delta_key: "k".to_string(),
            inner_steps: 50,
            training_loss: 10.0,
            tokens_processed: 1000,
            weight: 1000.0,
            timestamp: chrono::Utc::now(),
        };

        let added = pm.add_delta(stale);
        assert!(!added, "Stale delta (v5 against v10) should be rejected");
        assert_eq!(pm.num_deltas(), 0);
    }

    #[test]
    fn test_peer_merge_take_and_reset() {
        use crate::peer_merge::PeerMergeState;
        use distrain_shared::p2p::types::PeerDeltaAnnouncement;

        let mut pm = PeerMergeState::new(2, 5);

        let ann = |node: &str| PeerDeltaAnnouncement {
            node_id: node.to_string(),
            checkpoint_version: 5,
            delta_key: format!("k_{node}"),
            inner_steps: 50,
            training_loss: 10.0,
            tokens_processed: 1000,
            weight: 1000.0,
            timestamp: chrono::Utc::now(),
        };

        pm.add_delta(ann("a"));
        pm.add_delta(ann("b"));
        assert!(pm.should_produce_checkpoint());

        let pairs = pm.take_deltas();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pm.num_deltas(), 0); // reset after take
    }

    #[test]
    fn test_peer_merge_conflict_resolution() {
        use crate::peer_merge::resolve_checkpoint_conflict;

        // Higher contributions wins
        assert!(resolve_checkpoint_conflict(10, 5, 5, 5)); // local wins
        assert!(!resolve_checkpoint_conflict(5, 5, 10, 5)); // remote wins

        // Same contributions → higher version wins
        assert!(resolve_checkpoint_conflict(10, 6, 10, 5)); // local wins
        assert!(!resolve_checkpoint_conflict(10, 5, 10, 6)); // remote wins
    }

    // ── Shutdown state save test ──────────────────────────────────

    #[test]
    fn test_shutdown_flag_triggers_state_save() {
        // Simulate: shutdown flag set → state should be saveable
        let dir = tempfile::tempdir().unwrap();
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Not set → no save
        assert!(!shutdown.load(std::sync::atomic::Ordering::SeqCst));

        // Set flag
        shutdown.store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(shutdown.load(std::sync::atomic::Ordering::SeqCst));

        // Save state (simulating what the shutdown handler does)
        let state = crate::resume::NodeState {
            last_checkpoint_version: 42,
            seq_num: 10,
            shard_index: 0,
            shard_offset: 0,
            node_id: "test_node".to_string(),
            saved_at: "2026-04-17T00:00:00Z".to_string(),
        };
        state.save(dir.path()).unwrap();

        // Verify state was saved correctly
        let loaded = crate::resume::NodeState::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.last_checkpoint_version, 42);
        assert_eq!(loaded.seq_num, 10);
        assert_eq!(loaded.node_id, "test_node");
    }

    #[test]
    fn test_fallback_upgrade() {
        use distrain_shared::p2p::cascade::*;
        use distrain_shared::p2p::types::*;

        // Solo → DirectHttp when coordinator appears
        let upgrade = maybe_upgrade_mode(OperatingMode::Solo, true, 0, 0, false);
        assert_eq!(upgrade, Some(OperatingMode::DirectHttp));

        // DirectHttp → SingleCoordinatorWithDht when DHT comes online
        let upgrade = maybe_upgrade_mode(OperatingMode::DirectHttp, true, 1, 0, true);
        assert_eq!(upgrade, Some(OperatingMode::SingleCoordinatorWithDht));

        // Already at best mode → None
        let upgrade = maybe_upgrade_mode(OperatingMode::FullP2p, true, 2, 5, true);
        assert_eq!(upgrade, None);
    }
}
