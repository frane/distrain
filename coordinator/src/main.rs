//! Distrain Coordinator — stateless API server.
//!
//! 5 endpoints. All persistent state lives in R2.
//! Multiple instances can run concurrently (CRDT guarantees convergence).

pub mod aggregation;
pub mod metrics;
mod routes;
pub mod stats;
mod state;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Result;
use axum::extract::DefaultBodyLimit;
use axum::Router;
use distrain_shared::config::CoordinatorConfig;
use distrain_shared::storage::Storage;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

/// Shared application state passed to all handlers.
pub struct AppState {
    pub storage: Storage,
    pub config: CoordinatorConfig,
    /// Prevents concurrent aggregation runs.
    pub aggregation_in_progress: AtomicBool,
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

    let state = Arc::new(AppState {
        storage,
        config: config.clone(),
        aggregation_in_progress: AtomicBool::new(false),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .merge(routes::router())
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024)) // 512 MB for delta uploads
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

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

    config
}
