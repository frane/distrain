//! HTTP client for coordinator API.

use anyhow::{Context, Result};
use distrain_shared::config::NodeConfig;
use distrain_shared::types::*;

#[derive(Clone)]
pub struct CoordinatorClient {
    base_url: String,
    http: reqwest::Client,
}

impl CoordinatorClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::builder()
                .local_address(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED))
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Register this node with the coordinator.
    pub async fn register(
        &self,
        _config: &NodeConfig,
        persistent_node_id: Option<String>,
        hardware: Option<HardwareProfile>,
    ) -> Result<RegisterResponse> {
        let req = RegisterRequest {
            gpu_model: hardware.as_ref().map(|h| h.gpu_model.clone()).unwrap_or_else(|| "auto-detected".to_string()),
            gpu_memory_gb: hardware.as_ref().map(|h| h.vram_mb as f64 / 1024.0).unwrap_or(0.0),
            bandwidth_mbps: 0.0,
            node_id: persistent_node_id,
            hardware,
        };

        let resp = self
            .http
            .post(format!("{}/nodes/register", self.base_url))
            .json(&req)
            .send()
            .await
            .context("Failed to connect to coordinator")?;

        resp.json::<RegisterResponse>()
            .await
            .context("Failed to parse registration response")
    }

    /// Push a delta to the coordinator.
    pub async fn push_delta(&self, push: &DeltaPush) -> Result<DeltaPushResponse> {
        let resp = self
            .http
            .post(format!("{}/delta", self.base_url))
            .json(push)
            .send()
            .await
            .context("Failed to push delta")?;

        resp.json::<DeltaPushResponse>()
            .await
            .context("Failed to parse delta push response")
    }

    /// Get latest checkpoint info.
    pub async fn get_latest_checkpoint(&self) -> Result<CheckpointInfo> {
        let resp = self
            .http
            .get(format!("{}/checkpoint/latest", self.base_url))
            .send()
            .await
            .context("Failed to get checkpoint info")?;

        resp.json::<CheckpointInfo>()
            .await
            .context("Failed to parse checkpoint info")
    }

    /// Send heartbeat to coordinator with optional step progress.
    pub async fn heartbeat(
        &self,
        node_id: &str,
        step: Option<u64>,
        total_steps: Option<u64>,
        loss: Option<f64>,
        checkpoint_version: Option<u64>,
    ) -> Result<HeartbeatResponse> {
        let req = HeartbeatRequest {
            node_id: NodeId(node_id.to_string()),
            step,
            total_steps,
            loss,
            checkpoint_version,
        };
        let resp = self
            .http
            .post(format!("{}/heartbeat", self.base_url))
            .json(&req)
            .send()
            .await
            .context("Failed to send heartbeat")?;
        resp.json::<HeartbeatResponse>()
            .await
            .context("Failed to parse heartbeat response")
    }

    /// Sync heartbeat for use inside blocking threads (progress callbacks).
    pub fn heartbeat_sync(
        &self,
        node_id: &str,
        step: Option<u64>,
        total_steps: Option<u64>,
        loss: Option<f64>,
        checkpoint_version: Option<u64>,
    ) -> Result<HeartbeatResponse> {
        let req = HeartbeatRequest {
            node_id: NodeId(node_id.to_string()),
            step,
            total_steps,
            loss,
            checkpoint_version,
        };
        let resp = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new())
            .post(format!("{}/heartbeat", self.base_url))
            .json(&req)
            .send()
            .context("Failed to send heartbeat")?;
        resp.json::<HeartbeatResponse>()
            .context("Failed to parse heartbeat response")
    }

    /// Get auto-discovery config from coordinator.
    pub async fn get_config(&self) -> Result<NodeAutoConfig> {
        let resp = self
            .http
            .get(format!("{}/config", self.base_url))
            .send()
            .await
            .context("Failed to get config from coordinator")?;

        resp.json::<NodeAutoConfig>()
            .await
            .context("Failed to parse auto-config response")
    }

    /// Fetch pending replay requests from the bulletin board.
    pub async fn get_replay_board(&self) -> Result<Vec<serde_json::Value>> {
        let resp = self
            .http
            .get(format!("{}/replay_board", self.base_url))
            .send()
            .await
            .context("Failed to get replay board")?;

        resp.json::<Vec<serde_json::Value>>()
            .await
            .context("Failed to parse replay board")
    }

    /// Get training status.
    pub async fn get_status(&self) -> Result<TrainingStatus> {
        let resp = self
            .http
            .get(format!("{}/status", self.base_url))
            .send()
            .await
            .context("Failed to get status")?;

        resp.json::<TrainingStatus>()
            .await
            .context("Failed to parse training status")
    }
}
