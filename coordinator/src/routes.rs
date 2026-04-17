//! HTTP route handlers — 7 endpoints.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post, put};
use axum::{Json, Router};
use chrono::Utc;
use distrain_shared::types::*;
use sha2::{Digest, Sha256};
use tracing::{info, warn};
use uuid::Uuid;

use crate::metrics::METRICS;
use crate::state;
use crate::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/nodes/register", post(register_node))
        .route("/delta", post(push_delta))
        .route("/upload/*key", put(upload_to_storage))
        .route("/download/*key", get(download_from_storage))
        .route("/checkpoint/latest", get(get_latest_checkpoint))
        .route("/config", get(get_config))
        .route("/status", get(get_status))
        .route("/heartbeat", post(heartbeat))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
}

/// POST /nodes/register — register a new node.
async fn register_node(
    State(app): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    let node_id = if let Some(ref id) = req.node_id {
        NodeId(id.clone())
    } else {
        NodeId(format!("node_{}", Uuid::new_v4().as_simple()))
    };
    let api_key = Uuid::new_v4().to_string();
    let api_key_hash = format!("{:x}", Sha256::digest(api_key.as_bytes()));

    let registration = NodeRegistration {
        node_id: node_id.clone(),
        gpu_model: req.gpu_model,
        gpu_memory_gb: req.gpu_memory_gb,
        bandwidth_mbps: req.bandwidth_mbps,
        registered_at: Utc::now(),
        status: NodeStatus::Active,
        api_key_hash,
        last_seen: Utc::now(),
        total_contributions: 0,
    };

    // Add to registry
    let mut registry = state::load_registry(&app.storage).await.unwrap_or_default();
    registry.push(registration);

    METRICS
        .registered_nodes
        .store(registry.len() as u64, Ordering::Relaxed);
    METRICS.active_nodes.store(
        registry.iter().filter(|n| n.status == NodeStatus::Active).count() as u64,
        Ordering::Relaxed,
    );

    if let Err(e) = state::save_registry(&app.storage, &registry).await {
        warn!("Failed to save registry: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "Failed to save registration"})),
        );
    }

    info!("Registered node: {}", node_id);

    // Store hardware profile for capability-based checkpoint triggers
    if let Some(ref hw) = req.hardware {
        info!(
            "Node {} hardware: {} ({:?}, {} MiB VRAM, {} cores, {} MiB RAM)",
            node_id, hw.gpu_model, hw.device_type, hw.vram_mb, hw.cpu_cores, hw.ram_mb
        );
        {
            let mut coord_state = app.coord_state.write().await;
            coord_state.node_profiles.insert(node_id.clone().0, state::NodeProfile {
                device_type: hw.device_type.clone(),
                vram_mb: hw.vram_mb,
                gpu_model: hw.gpu_model.clone(),
                round_time_secs: None,
                expected_round_time: hw.expected_round_time_secs,
                step_time_secs: hw.step_time_secs,
                h_mini: hw.h_mini,
                last_push_time: None,
            });
            if let Some(ert) = hw.expected_round_time_secs {
                info!("Node {} calibrated: step_time={:.2}s, H_mini={}, expected_round={:.0}s",
                    node_id, hw.step_time_secs.unwrap_or(0.0), hw.h_mini.unwrap_or(0), ert);
            }
        } // write lock released
    }

    let node_endpoint = app.config.external_storage_endpoint
        .as_deref()
        .unwrap_or(&app.config.storage.endpoint)
        .to_string();

    (
        StatusCode::OK,
        Json(
            serde_json::to_value(RegisterResponse {
                node_id,
                api_key,
                status: NodeStatus::Active,
                storage_endpoint: Some(node_endpoint),
                storage_bucket: Some(app.config.storage.bucket.clone()),
                training_params: Some(TrainingParams::default()),
            })
            .unwrap(),
        ),
    )
}

/// PUT /upload/{key} — proxy upload to S3 (for browser nodes that can't PUT directly to S3 due to CORS).
async fn upload_to_storage(
    State(app): State<Arc<AppState>>,
    Path(key): Path<String>,
    body: Bytes,
) -> impl IntoResponse {
    let size = body.len();
    match app.storage.put(&key, body.to_vec()).await {
        Ok(()) => {
            info!("Proxied upload: {key} ({size} bytes)");
            StatusCode::OK
        }
        Err(e) => {
            warn!("Upload proxy failed for {key}: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

/// GET /download/{key} — proxy download from S3 (for browser nodes that can't GET from S3 due to CORS).
async fn download_from_storage(
    State(app): State<Arc<AppState>>,
    Path(key): Path<String>,
) -> impl IntoResponse {
    match app.storage.get(&key).await {
        Ok(data) => {
            info!("Proxied download: {key} ({} bytes)", data.len());
            (StatusCode::OK, data).into_response()
        }
        Err(e) => {
            warn!("Download proxy failed for {key}: {e}");
            (StatusCode::NOT_FOUND, format!("Not found: {key}")).into_response()
        }
    }
}

/// POST /delta — push a delta.
async fn push_delta(
    State(app): State<Arc<AppState>>,
    Json(push): Json<DeltaPush>,
) -> impl IntoResponse {
    METRICS
        .delta_pushes_total
        .fetch_add(1, Ordering::Relaxed);

    // Apply delta to in-memory accumulator
    let (accepted, reason, staleness, weight, current_version, num_contribs);
    {
        let mut acc = app.accumulator.write().await;
        let result = state::apply_delta_push(
            &mut acc,
            &push,
            app.config.staleness_decay,
            app.config.max_staleness,
        );
        accepted = result.0;
        reason = result.1;

        if accepted {
            staleness = acc.checkpoint_version.saturating_sub(push.checkpoint_version);
            weight = acc.contributions.last().map(|c| c.weight).unwrap_or(0.0);
        } else {
            staleness = 0;
            weight = 0.0;
        }

        num_contribs = acc.contributions.len();
        current_version = acc.checkpoint_version;
    } // write lock released

    if accepted {
        METRICS
            .delta_pushes_accepted
            .fetch_add(1, Ordering::Relaxed);
        info!(
            "Accepted delta from {} (seq={}, steps={}, staleness={})",
            push.node_id, push.seq_num, push.inner_steps, staleness,
        );

        // Best-effort stats append
        let stats_storage = app.storage.clone();
        let stats_entry = crate::stats::DeltaAcceptedEntry {
            event: "delta_accepted",
            timestamp: Utc::now(),
            node_id: push.node_id.clone(),
            seq_num: push.seq_num,
            checkpoint_version: push.checkpoint_version,
            inner_steps: push.inner_steps,
            training_loss: push.training_loss,
            weight,
            staleness,
            tokens_processed: Some(push.tokens_processed),
            compressed_bytes: push.compressed_bytes,
            dense_norm: push.dense_norm,
            sparse_norm: push.sparse_norm,
        };
        tokio::spawn(async move {
            crate::stats::append_stats_entry(&stats_storage, &stats_entry).await;
        });
    } else {
        METRICS
            .delta_pushes_rejected
            .fetch_add(1, Ordering::Relaxed);
        info!(
            "Rejected delta from {}: {}",
            push.node_id,
            reason.as_deref().unwrap_or("unknown")
        );

        // Post to replay board if rejected due to extreme staleness (13+)
        let staleness_val = current_version.saturating_sub(push.checkpoint_version);
        if staleness_val >= 13 {
            let mut request = crate::proxy_replay::create_replay_request(
                &push.node_id.0,
                current_version,
                staleness_val,
                push.training_loss,
                push.inner_steps,
                4, // expire after 4 hours
            );
            if let Some(ref shard_ids) = push.shard_ids {
                request.shard_ids = shard_ids.clone();
            }
            let replay_storage = app.storage.clone();
            tokio::spawn(async move {
                crate::proxy_replay::post_replay_request(&replay_storage, &request).await;
            });
        }
    }

    METRICS
        .accumulator_contributions
        .store(num_contribs as u64, Ordering::Relaxed);

    // Track round time in coordinator state (in-memory)
    if accepted {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut coord_state = app.coord_state.write().await;
        if let Some(profile) = coord_state.node_profiles.get_mut(&push.node_id.0) {
            if let Some(last_push) = profile.last_push_time {
                let elapsed = now_secs.saturating_sub(last_push) as f64;
                if elapsed > 10.0 && elapsed < 600.0 {
                    profile.round_time_secs = Some(match profile.round_time_secs {
                        Some(prev) => prev * 0.3 + elapsed * 0.7,
                        None => elapsed,
                    });
                }
            }
            profile.last_push_time = Some(now_secs);
        }
    } // write lock released

    // Checkpoint trigger check — read locks only
    let (should_ckpt, acc_snapshot, coord_snapshot);
    {
        let acc = app.accumulator.read().await;
        let coord_state = app.coord_state.read().await;
        should_ckpt = state::should_checkpoint(&acc, app.config.min_contributions, &coord_state);
        // Clone for aggregation spawn (before releasing locks)
        acc_snapshot = acc.clone();
        coord_snapshot = coord_state.clone();
    } // read locks released

    let agg_in_progress = app.aggregation_in_progress.load(Ordering::SeqCst);
    if accepted {
        let active_count = coord_snapshot.heartbeats.len();
        let first_received = acc_snapshot.contributions.iter().map(|c| c.received_at).min();
        let wait_secs = first_received.map(|f| (chrono::Utc::now() - f).num_seconds()).unwrap_or(0);
        info!(
            "Checkpoint check: contributions={}/{} active, waiting={}s, should_checkpoint={}, agg_in_progress={}",
            acc_snapshot.contributions.len(), active_count, wait_secs, should_ckpt, agg_in_progress,
        );
    }
    if accepted && should_ckpt && !agg_in_progress
    {
        // Try to acquire the aggregation lock
        if app
            .aggregation_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            info!(
                "Triggering aggregation: {} contributions (active_nodes={})",
                acc_snapshot.contributions.len(), coord_snapshot.active_nodes
            );

            METRICS.aggregations_total.fetch_add(1, Ordering::Relaxed);

            // Flush accumulator to S3 before aggregation (important state)
            if let Err(e) = state::save_accumulator(&app.storage, &acc_snapshot).await {
                warn!("Failed to save accumulator before aggregation: {e}");
            }

            // Spawn aggregation with cloned data (no locks held)
            let storage = app.storage.clone();
            let outer_lr = app.config.outer_lr;
            let outer_momentum = app.config.outer_momentum;
            let keep_versions = app.config.keep_versions;
            let vocab_size = app.config.vocab_size;
            let enable_rebasing = app.config.enable_rebasing;
            let rebasing_threshold = app.config.rebasing_threshold;
            let rebasing_coefficient = app.config.rebasing_coefficient;
            let agg_flag = Arc::clone(&app);

            tokio::spawn(async move {
                let agg_start = std::time::Instant::now();
                match crate::aggregation::run_aggregation(
                    &storage,
                    &acc_snapshot,
                    outer_lr,
                    outer_momentum,
                    keep_versions,
                    vocab_size,
                    enable_rebasing,
                    rebasing_threshold,
                    rebasing_coefficient,
                )
                .await
                {
                    Ok((new_version, tokens_this_checkpoint, contributor_ids)) => {
                        let agg_secs = agg_start.elapsed().as_secs_f64();
                        info!("Checkpoint v{new_version} produced in {agg_secs:.1}s");
                        METRICS
                            .checkpoint_version
                            .store(new_version, Ordering::Relaxed);
                        METRICS
                            .accumulator_contributions
                            .store(0, Ordering::Relaxed);
                        *METRICS.last_aggregation_secs.lock().unwrap() = agg_secs;

                        // Reset in-memory accumulator
                        {
                            let mut acc = agg_flag.accumulator.write().await;
                            *acc = AccumulatorState {
                                checkpoint_version: new_version,
                                contributions: Vec::new(),
                                first_contribution_at: None,
                                version: acc.version + 1,
                            };
                        }

                        // Update in-memory coordinator state
                        {
                            let mut cs = agg_flag.coord_state.write().await;
                            cs.recent_contributors.push((new_version, contributor_ids));
                            if cs.recent_contributors.len() > 5 {
                                let excess = cs.recent_contributors.len() - 5;
                                cs.recent_contributors.drain(..excess);
                            }
                            let recent_window = cs.recent_contributors.len().saturating_sub(3);
                            let mut all_nodes: std::collections::HashSet<&str> = std::collections::HashSet::new();
                            for (_, nodes) in &cs.recent_contributors[recent_window..] {
                                for n in nodes {
                                    all_nodes.insert(n);
                                }
                            }
                            cs.active_nodes = all_nodes.len() as u64;
                            cs.total_tokens_trained += tokens_this_checkpoint;
                        }

                        // Immediate flush all state to R2 after checkpoint production
                        let acc_snap = agg_flag.accumulator.read().await.clone();
                        let cs_snap = agg_flag.coord_state.read().await.clone();
                        if let Err(e) =
                            state::save_state_to_r2(&storage, &acc_snap, &cs_snap).await
                        {
                            warn!("Post-aggregation flush failed: {e}");
                        }

                        // Broadcast checkpoint via P2P gossip (if enabled)
                        if let Some(ref p2p) = agg_flag.p2p {
                            let announcement = distrain_shared::p2p::types::CheckpointAnnouncement {
                                version: new_version,
                                r2_path: distrain_shared::paths::checkpoint_path(new_version),
                                delta_path: Some(distrain_shared::paths::checkpoint_delta_path(
                                    new_version, new_version - 1,
                                )),
                                loss: cs_snap.total_tokens_trained as f64, // placeholder
                                total_tokens: cs_snap.total_tokens_trained,
                                timestamp: chrono::Utc::now(),
                                produced_by: p2p.local_peer_id.clone(),
                                num_contributions: acc_snapshot.contributions.len() as u64,
                            };
                            if let Err(e) = p2p.announce_checkpoint(announcement).await {
                                warn!("P2P checkpoint broadcast failed: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Aggregation failed: {e}");
                        METRICS.aggregations_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                // Release aggregation lock
                agg_flag
                    .aggregation_in_progress
                    .store(false, Ordering::SeqCst);
            });
        }
    }

    (
        StatusCode::OK,
        Json(
            serde_json::to_value(DeltaPushResponse {
                accepted,
                checkpoint_version: current_version,
                reason,
            })
            .unwrap(),
        ),
    )
}

/// GET /checkpoint/latest — current checkpoint info.
async fn get_latest_checkpoint(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let version = app.accumulator.read().await.checkpoint_version;
    let delta_key = if version > 0 {
        Some(distrain_shared::paths::checkpoint_delta_path(version, version - 1))
    } else {
        None
    };
    let info = CheckpointInfo {
        version,
        checkpoint_key: distrain_shared::paths::checkpoint_path(version),
        metadata_key: distrain_shared::paths::checkpoint_metadata_path(version),
        val_loss: None,
        total_contributions: 0,
        total_tokens: 0,
        created_at: Utc::now(),
        delta_key,
        delta_from_version: if version > 0 { Some(version - 1) } else { None },
    };

    Json(info)
}

/// GET /config — auto-discovery: returns everything a node needs to join.
async fn get_config(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    // Return external storage endpoint if configured (for Docker deployments where
    // coordinator uses localhost:9000 internally but nodes need the public URL).
    let node_endpoint = app.config.external_storage_endpoint
        .as_deref()
        .unwrap_or(&app.config.storage.endpoint)
        .to_string();

    let auto_config = NodeAutoConfig {
        storage: StorageConfigPublic {
            endpoint: node_endpoint,
            bucket: app.config.storage.bucket.clone(),
            access_key_id: app.config.storage.access_key_id.clone(),
            secret_access_key: app.config.storage.secret_access_key.clone(),
            region: app.config.storage.region.clone(),
        },
        training_params: TrainingParams::default(),
        coordinator_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    Json(auto_config)
}

/// GET /status — public training status.
async fn get_status(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let acc = app.accumulator.read().await;
    let coord_state = app.coord_state.read().await;

    // Active nodes from heartbeat TTL (30 min)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let ttl = 30 * 60;
    let active_nodes = coord_state.heartbeats.values()
        .filter(|ts| now.saturating_sub(**ts) < ttl)
        .count() as u64;

    // Weighted average loss from current contributions
    let total_w: f64 = acc.contributions.iter().map(|c| c.weight).sum();
    let latest_val_loss = if total_w > 0.0 {
        let avg = acc.contributions.iter().map(|c| c.training_loss * c.weight).sum::<f64>() / total_w;
        if avg > 0.0 { Some(avg) } else { None }
    } else {
        None
    };

    let status = TrainingStatus {
        checkpoint_version: acc.checkpoint_version,
        active_nodes,
        total_contributions: acc.version,
        total_tokens_trained: coord_state.total_tokens_trained,
        accumulator_contributions: acc.contributions.len() as u64,
        latest_val_loss,
        loss_history: Vec::new(),
        node_last_seen: coord_state.heartbeats.iter()
            .map(|(id, ts)| (id.clone(), *ts))
            .collect(),
    };

    Json(status)
}

/// POST /heartbeat — node liveness signal with step progress.
async fn heartbeat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<HeartbeatRequest>,
) -> impl IntoResponse {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let active = {
        let mut coord_state = app.coord_state.write().await;
        coord_state.heartbeats.insert(req.node_id.0.clone(), now);

        // Expire nodes not seen in 10 minutes (tighter TTL since heartbeats are per-step now)
        let ttl = 10 * 60;
        coord_state.heartbeats.retain(|_, ts| now.saturating_sub(*ts) < ttl);
        coord_state.active_nodes = coord_state.heartbeats.len() as u64;
        coord_state.active_nodes
    }; // write lock released

    METRICS.active_nodes.store(active, Ordering::Relaxed);

    // Check if node should abort: if checkpoint advanced beyond what it's training on
    let current_version = app.accumulator.read().await.checkpoint_version;
    let should_abort = if let Some(node_version) = req.checkpoint_version {
        let staleness = current_version.saturating_sub(node_version);
        staleness >= 3 // abort if 3+ versions behind
    } else {
        false
    };

    Json(HeartbeatResponse {
        active_nodes: active,
        should_abort,
        latest_version: Some(current_version),
    })
}

/// GET /health — health check.
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

/// GET /metrics — Prometheus metrics.
async fn metrics() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        METRICS.render(),
    )
}
