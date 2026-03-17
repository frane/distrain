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
    pub async fn register(&self, _config: &NodeConfig, persistent_node_id: Option<String>) -> Result<RegisterResponse> {
        let req = RegisterRequest {
            gpu_model: "auto-detected".to_string(),
            gpu_memory_gb: 0.0,
            bandwidth_mbps: 0.0,
            node_id: persistent_node_id,
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

    /// Send heartbeat to coordinator.
    pub async fn heartbeat(&self, node_id: &str) -> Result<HeartbeatResponse> {
        let req = HeartbeatRequest {
            node_id: NodeId(node_id.to_string()),
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
