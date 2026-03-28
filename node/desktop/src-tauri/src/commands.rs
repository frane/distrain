//! Tauri IPC commands — bridge between frontend and Burn training node.
//!
//! Uses wgpu backend (Metal on macOS, Vulkan on Linux/Windows) for GPU training.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use tauri::State;
use tokio::sync::Mutex;
use tracing::info;

use distrain_node::{client, data, trainer};
use distrain_shared::config::NodeConfig;
use distrain_shared::storage::Storage;

/// Shared application state accessible from all commands.
pub struct AppState {
    pub training_active: Mutex<bool>,
    /// Stats and logs use std::sync::Mutex so the GPU training thread's
    /// progress callback can update them without async.
    pub stats: std::sync::Mutex<TrainingStats>,
    pub logs: std::sync::Mutex<Vec<String>>,
    pub config: Mutex<NodeConfig>,
    pub stop_signal: AtomicBool,
    pub node_info: Mutex<NodeInfo>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            training_active: Mutex::new(false),
            stats: std::sync::Mutex::new(TrainingStats::default()),
            logs: std::sync::Mutex::new(Vec::new()),
            config: Mutex::new(NodeConfig::default()),
            stop_signal: AtomicBool::new(false),
            node_info: Mutex::new(NodeInfo::default()),
        }
    }
}

/// Helper to push a log message to the UI queue.
async fn push_log(state: &Arc<AppState>, msg: String) {
    info!("{msg}");
    let mut logs = state.logs.lock().unwrap();
    logs.push(msg);
    if logs.len() > 500 {
        let drain = logs.len() - 500;
        logs.drain(..drain);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingStats {
    pub current_step: u64,
    pub total_steps: u64,
    pub current_loss: f64,
    pub tokens_processed: u64,
    pub tokens_per_sec: f64,
    pub elapsed_secs: f64,
    pub checkpoint_version: u64,
    pub rounds_completed: u64,
    pub delta_upload_size_bytes: u64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct NodeInfo {
    pub node_id: Option<String>,
    pub h_mini: Option<u64>,
    pub connected: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeStatus {
    pub is_training: bool,
    pub is_calibrated: bool,
    pub node_id: Option<String>,
    pub gpu_info: String,
    pub h_mini: Option<u64>,
    pub connected: bool,
}

/// Get current node status.
#[tauri::command]
pub async fn get_node_status(state: State<'_, Arc<AppState>>) -> Result<NodeStatus, String> {
    let active = *state.training_active.lock().await;
    let node_info = state.node_info.lock().await;

    Ok(NodeStatus {
        is_training: active,
        is_calibrated: node_info.h_mini.is_some(),
        node_id: node_info.node_id.clone(),
        gpu_info: "GPU (wgpu/Metal backend)".to_string(),
        h_mini: node_info.h_mini,
        connected: node_info.connected,
    })
}

/// Start the training loop in the background.
#[tauri::command]
pub async fn start_training(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    {
        let active = state.training_active.lock().await;
        if *active {
            return Err("Training is already running".to_string());
        }
    }

    let state_clone = state.inner().clone();
    let config = state.config.lock().await.clone();

    state.stop_signal.store(false, Ordering::SeqCst);
    *state.training_active.lock().await = true;

    tokio::spawn(async move {
        if let Err(e) = run_training_loop(state_clone.clone(), config).await {
            push_log(&state_clone, format!("Training error: {e:#}")).await;
        }
        *state_clone.training_active.lock().await = false;
        push_log(&state_clone, "Training stopped".to_string()).await;
    });

    Ok("Training started on GPU (wgpu)".to_string())
}

/// Background training loop — mirrors the CLI node's loop.
async fn run_training_loop(state: Arc<AppState>, mut config: NodeConfig) -> anyhow::Result<()> {
    let coordinator = client::CoordinatorClient::new(&config.coordinator_url);
    let storage = Storage::new(&config.storage).await?;

    // Register (desktop doesn't persist node ID yet)
    let reg = coordinator.register(&config, None, None).await?;
    let node_id = reg.node_id.0.clone();
    push_log(&state, format!("Registered as {node_id}")).await;

    // Merge coordinator training params (coordinator is source of truth)
    if let Some(params) = reg.training_params {
        push_log(
            &state,
            format!(
                "Training params: lr={:.2e}→{:.2e}, warmup={:.0}%, grad_clip={}, shards={:.0}%",
                params.lr_max, params.lr_min, params.warmup_fraction * 100.0,
                params.grad_clip_norm, params.shards_fraction * 100.0,
            ),
        )
        .await;
        config.training_params = Some(params.clone());
        let mut cfg = state.config.lock().await;
        cfg.training_params = Some(params);
    }

    {
        let mut info = state.node_info.lock().await;
        info.node_id = Some(node_id.clone());
        info.connected = true;
    }

    let mut seq_num: u64 = 0;

    // Calibrate
    let (h_mini, _use_cpu) = trainer::calibrate(&config).await?;
    push_log(&state, format!("Calibrated: H_mini = {h_mini}")).await;

    {
        let mut info = state.node_info.lock().await;
        info.h_mini = Some(h_mini);
    }

    // Cache dir
    let cache_dir = shellexpand::tilde(&config.cache_dir).to_string();
    let cache_dir = std::path::PathBuf::from(cache_dir);
    tokio::fs::create_dir_all(&cache_dir).await?;

    // Load data manifest (shard list — actual data loaded per round via shard assignment)
    let (manifest, data_cache) = data::DataLoader::load_manifest(&storage, &cache_dir).await?;
    let total_shards = manifest.shards.len();
    let params = config
        .training_params
        .as_ref()
        .cloned()
        .unwrap_or_default();
    let shards_per_node = params.shards_per_node(total_shards);
    push_log(&state, format!("Data manifest: {total_shards} shards")).await;

    loop {
        if state.stop_signal.load(Ordering::SeqCst) {
            break;
        }

        let ckpt_info = coordinator.get_latest_checkpoint().await?;
        let version = ckpt_info.version;

        // Deterministic shard assignment
        let shard_ids = distrain_model::compute_shard_assignment(
            &node_id, version, total_shards, shards_per_node,
        );
        let mut data_loader = data::DataLoader::from_assignment(
            &storage, &manifest, &data_cache, &shard_ids,
            config.seq_len, config.batch_size,
        ).await?;

        push_log(&state, format!(
            "v{version}: {h_mini} steps, {} shards, {} tokens",
            shard_ids.len(), data_loader.total_tokens()
        )).await;

        // Download checkpoint if not cached
        let ckpt_path = cache_dir.join(format!("v{version}_model.safetensors"));
        if !ckpt_path.exists() {
            storage
                .download_to_file(&ckpt_info.checkpoint_key, &ckpt_path)
                .await?;
        }

        // Update checkpoint version in stats
        {
            let mut stats = state.stats.lock().unwrap();
            stats.checkpoint_version = version;
        }

        // Poll for checkpoint advancement in background
        let ckpt_abort = std::sync::Arc::new(AtomicBool::new(false));
        let ckpt_abort_bg = ckpt_abort.clone();
        let coordinator_bg = coordinator.clone();
        let bg_version = version;
        let poll_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                if let Ok(info) = coordinator_bg.get_latest_checkpoint().await {
                    if info.version > bg_version {
                        ckpt_abort_bg.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
        });

        // Train with per-step progress (aborts on stop signal or new checkpoint)
        seq_num += 1;
        let delta_path = cache_dir.join(format!("delta_{node_id}_{seq_num}.delta.zst"));
        let state_for_progress = state.clone();
        let ckpt_abort_cb = ckpt_abort.clone();
        let round_base_tokens = state.stats.lock().unwrap().tokens_processed;
        let mut error_buffer = distrain_model::compression::ErrorBuffer::new();
        let result = trainer::run_training_with_progress(
            &config,
            &ckpt_path,
            h_mini,
            &delta_path,
            &mut data_loader,
            &mut error_buffer,
            config.batch_size,
            1, // grad_accum_steps — desktop uses default batch_size for now
            move |progress: trainer::StepProgress| {
                let mut stats = state_for_progress.stats.lock().unwrap();
                stats.current_step = progress.step;
                stats.total_steps = progress.total_steps;
                stats.current_loss = progress.loss;
                stats.tokens_processed = round_base_tokens + progress.tokens_processed;
                stats.elapsed_secs = progress.elapsed_secs;
                if progress.elapsed_secs > 0.0 {
                    stats.tokens_per_sec =
                        progress.tokens_processed as f64 / progress.elapsed_secs;
                }
                if progress.step % 10 == 0 {
                    let mut logs = state_for_progress.logs.lock().unwrap();
                    logs.push(format!(
                        "Step {}/{}: loss={:.4}, lr={:.2e}",
                        progress.step, progress.total_steps, progress.loss, progress.lr
                    ));
                }
                // Abort if stop requested or checkpoint advanced
                state_for_progress.stop_signal.load(Ordering::SeqCst)
                    || ckpt_abort_cb.load(Ordering::SeqCst)
            },
        )
        .await?;
        poll_handle.abort();

        // Update stats after round
        {
            let mut stats = state.stats.lock().unwrap();
            stats.current_loss = result.final_loss;
            stats.tokens_processed = round_base_tokens + result.tokens_processed;
            stats.elapsed_secs += result.elapsed_secs;
            stats.rounds_completed += 1;
            stats.current_step = stats.total_steps;
            if result.elapsed_secs > 0.0 {
                stats.tokens_per_sec =
                    result.tokens_processed as f64 / result.elapsed_secs;
            }
        }

        push_log(
            &state,
            format!(
                "Round done: {h_mini} steps, loss={:.4}, {:.1}s",
                result.final_loss, result.elapsed_secs
            ),
        )
        .await;

        // Upload delta
        let delta_key = distrain_shared::paths::delta_path(version, &node_id, seq_num);
        storage.upload_from_file(&delta_key, &delta_path).await?;

        // Push metadata
        let push = distrain_shared::types::DeltaPush {
            node_id: distrain_shared::types::NodeId(node_id.clone()),
            seq_num,
            checkpoint_version: version,
            inner_steps: h_mini,
            delta_key,
            training_loss: result.final_loss,
            tokens_processed: result.tokens_processed,
            training_time_secs: result.elapsed_secs,
        };

        let resp = coordinator.push_delta(&push).await?;
        if resp.accepted {
            push_log(
                &state,
                format!("Push accepted (v{})", resp.checkpoint_version),
            )
            .await;
        } else {
            push_log(
                &state,
                format!(
                    "Delta rejected: {}",
                    resp.reason.unwrap_or_else(|| "unknown".to_string())
                ),
            )
            .await;
        }

        let _ = tokio::fs::remove_file(&delta_path).await;
    }

    Ok(())
}

/// Stop the training loop.
#[tauri::command]
pub async fn stop_training(state: State<'_, Arc<AppState>>) -> Result<String, String> {
    state.stop_signal.store(true, Ordering::SeqCst);
    Ok("Stopping...".to_string())
}

/// Run device calibration to determine optimal H_mini.
#[tauri::command]
pub async fn calibrate_device(state: State<'_, Arc<AppState>>) -> Result<u64, String> {
    let config = state.config.lock().await.clone();
    let (h_mini, _use_cpu) = trainer::calibrate(&config)
        .await
        .map_err(|e| format!("Calibration failed: {e:#}"))?;
    Ok(h_mini)
}

/// Get real-time training statistics.
#[tauri::command]
pub async fn get_training_stats(state: State<'_, Arc<AppState>>) -> Result<TrainingStats, String> {
    let stats = state.stats.lock().unwrap();
    Ok(stats.clone())
}

/// Get new log messages since last poll (drains the queue).
#[tauri::command]
pub async fn get_logs(state: State<'_, Arc<AppState>>) -> Result<Vec<String>, String> {
    let mut logs = state.logs.lock().unwrap();
    let result = logs.drain(..).collect();
    Ok(result)
}

/// Get current node configuration.
#[tauri::command]
pub async fn get_config(state: State<'_, Arc<AppState>>) -> Result<NodeConfig, String> {
    let config = state.config.lock().await;
    Ok(config.clone())
}

/// Save node configuration.
#[tauri::command]
pub async fn save_config(
    state: State<'_, Arc<AppState>>,
    config: NodeConfig,
) -> Result<String, String> {
    let mut current = state.config.lock().await;
    *current = config;
    Ok("Configuration saved".to_string())
}

/// Get current training parameters.
#[tauri::command]
pub async fn get_training_params(
    state: State<'_, Arc<AppState>>,
) -> Result<distrain_shared::types::TrainingParams, String> {
    let config = state.config.lock().await;
    Ok(config.training_params.clone().unwrap_or_default())
}

/// Save training parameters.
#[tauri::command]
pub async fn save_training_params(
    state: State<'_, Arc<AppState>>,
    params: distrain_shared::types::TrainingParams,
) -> Result<String, String> {
    let mut config = state.config.lock().await;
    config.training_params = Some(params);
    Ok("Training params saved".to_string())
}
