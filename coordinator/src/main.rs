//! Distrain Coordinator — API server with in-memory state.
//!
//! Accumulator and coordinator state live in memory, flushed to R2 every 30s.
//! Checkpoint production flushes immediately.

pub mod aggregation;
pub mod metrics;
pub mod proxy_replay;
mod routes;
pub mod stats;
pub mod state;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::DefaultBodyLimit;
use axum::Router;
use distrain_shared::config::CoordinatorConfig;
use distrain_shared::storage::Storage;
use distrain_shared::types::AccumulatorState;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::state::CoordinatorPersistentState;

/// Shared application state passed to all handlers.
pub struct AppState {
    pub storage: Storage,
    pub config: CoordinatorConfig,
    /// Prevents concurrent aggregation runs.
    pub aggregation_in_progress: AtomicBool,
    /// In-memory accumulator state. Flushed to R2 periodically.
    pub accumulator: RwLock<AccumulatorState>,
    /// In-memory coordinator persistent state. Flushed to R2 periodically.
    pub coord_state: RwLock<CoordinatorPersistentState>,
    /// P2P service handle (if P2P is enabled). Used to broadcast checkpoint announcements.
    pub p2p: Option<distrain_shared::p2p::service::P2pHandle>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug".into()),
        )
        .init();

    let config = load_config();
    info!("Starting coordinator on {}:{}", config.host, config.port);
    info!(
        "Outer LR: initial={}, loss-based scheduling (vocab_size={}, ln_vocab={:.2})",
        config.outer_lr, config.vocab_size, (config.vocab_size as f64).ln()
    );
    info!(
        "Checkpoint trigger: min_contributions={}, min_weight={:.2}",
        config.min_contributions, config.min_weight
    );

    let storage = Storage::new(&config.storage).await?;
    storage.ensure_bucket().await?;

    // Recover all state from R2 (accumulator, coordinator state, registry)
    let recovered = state::recover_state_from_r2(&storage).await;
    let accumulator = recovered.accumulator;
    let coord_state = recovered.coord_state;

    // Start P2P service if enabled
    let p2p_handle = if config.p2p.enabled {
        match distrain_shared::p2p::service::start_p2p_service(&config.p2p).await {
            Ok((handle, mut event_rx)) => {
                info!("P2P service started: peer_id={}", handle.local_peer_id);
                // Register as coordinator on DHT
                let http_addr = format!("http://{}:{}", config.host, config.port);
                if let Err(e) = handle.register_coordinator(http_addr).await {
                    warn!("Failed to register on DHT: {e}");
                }
                // Spawn event handler (logs events for now)
                tokio::spawn(async move {
                    while let Some(event) = event_rx.recv().await {
                        match event {
                            distrain_shared::p2p::service::P2pEvent::PeerConnected(peer) => {
                                info!("P2P: peer connected: {}", &peer[..12.min(peer.len())]);
                            }
                            distrain_shared::p2p::service::P2pEvent::PeerDisconnected(peer) => {
                                info!("P2P: peer disconnected: {}", &peer[..12.min(peer.len())]);
                            }
                            distrain_shared::p2p::service::P2pEvent::CoordinatorSync(msg) => {
                                info!("P2P: coordinator sync from {} (v{})", msg.coordinator_id, msg.checkpoint_version);
                                // TODO: implement multi-coordinator merge (item 10.3)
                            }
                            _ => {}
                        }
                    }
                });
                Some(handle)
            }
            Err(e) => {
                warn!("Failed to start P2P service: {e}. Running in direct HTTP mode.");
                None
            }
        }
    } else {
        None
    };

    let app_state = Arc::new(AppState {
        storage,
        config: config.clone(),
        aggregation_in_progress: AtomicBool::new(false),
        accumulator: RwLock::new(accumulator),
        coord_state: RwLock::new(coord_state),
        p2p: p2p_handle,
    });

    // Spawn background task to persist state to R2 every 30 seconds
    {
        let app = Arc::clone(&app_state);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                // Clone under read locks, then write to S3 outside locks
                let acc_snapshot = app.accumulator.read().await.clone();
                let coord_snapshot = app.coord_state.read().await.clone();

                if let Err(e) =
                    state::save_state_to_r2(&app.storage, &acc_snapshot, &coord_snapshot).await
                {
                    warn!("Background flush failed: {e}");
                }
            }
        });
    }

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .merge(routes::router())
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024)) // 512 MB for delta uploads
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Listening on {addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

fn load_config() -> CoordinatorConfig {
    let mut config = CoordinatorConfig::default();

    if let Ok(v) = std::env::var("PORT") {
        config.port = v.parse().unwrap_or(8000);
    }
    if let Ok(v) = std::env::var("HOST") {
        config.host = v;
    }
    if let Ok(v) = std::env::var("R2_ENDPOINT") {
        config.storage.endpoint = v;
    }
    if let Ok(v) = std::env::var("R2_BUCKET") {
        config.storage.bucket = v;
    }
    if let Ok(v) = std::env::var("R2_ACCESS_KEY_ID") {
        config.storage.access_key_id = v;
    }
    if let Ok(v) = std::env::var("R2_SECRET_ACCESS_KEY") {
        config.storage.secret_access_key = v;
    }
    if let Ok(v) = std::env::var("MIN_CONTRIBUTIONS") {
        config.min_contributions = v.parse().unwrap_or(4);
    }
    if let Ok(v) = std::env::var("OUTER_LR") {
        config.outer_lr = v.parse().unwrap_or(0.1);
    }
    if let Ok(v) = std::env::var("OUTER_MOMENTUM") {
        config.outer_momentum = v.parse().unwrap_or(0.9);
    }
    if let Ok(v) = std::env::var("KEEP_VERSIONS") {
        config.keep_versions = v.parse().unwrap_or(10);
    }
    if let Ok(v) = std::env::var("MAX_STALENESS") {
        config.max_staleness = v.parse().unwrap_or(10);
    }
    if let Ok(v) = std::env::var("VOCAB_SIZE") {
        config.vocab_size = v.parse().unwrap_or(32768);
    }
    if let Ok(v) = std::env::var("MIN_WEIGHT") {
        config.min_weight = v.parse().unwrap_or(1.5);
    }
    if let Ok(v) = std::env::var("S3_EXTERNAL_ENDPOINT") {
        config.external_storage_endpoint = Some(v);
    }
    if let Ok(v) = std::env::var("ENABLE_REBASING") {
        config.enable_rebasing = v.parse().unwrap_or(true);
    }
    if let Ok(v) = std::env::var("REBASING_THRESHOLD") {
        config.rebasing_threshold = v.parse().unwrap_or(3);
    }
    if let Ok(v) = std::env::var("REBASING_COEFFICIENT") {
        config.rebasing_coefficient = v.parse().unwrap_or(0.5);
    }

    config
}
