//! P2P message types and configuration.
//!
//! These types are always available (no feature gate) so that both
//! coordinator and node can define their P2P interfaces without
//! pulling in libp2p as a hard dependency.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ── Configuration ──────────────────────────────────────────────────

/// P2P networking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2pConfig {
    /// Enable P2P networking (DHT + gossip). Default false.
    #[serde(default)]
    pub enabled: bool,
    /// Bootstrap peer addresses (multiaddr format).
    /// e.g., "/ip4/1.2.3.4/tcp/4001/p2p/QmPeer..."
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
    /// TCP port to listen on for P2P connections. 0 = random.
    #[serde(default = "default_listen_port")]
    pub listen_port: u16,
    /// Run ID for topic namespacing. Nodes in different runs don't see each other.
    #[serde(default = "default_run_id")]
    pub run_id: String,
    /// Role: "coordinator" or "node". Determines DHT registration behavior.
    #[serde(default = "default_role")]
    pub role: String,
    /// Enable mDNS for local network peer discovery. Default true.
    #[serde(default = "default_mdns")]
    pub mdns: bool,
}

fn default_listen_port() -> u16 { 0 }
fn default_run_id() -> String { "default".to_string() }
fn default_role() -> String { "node".to_string() }
fn default_mdns() -> bool { true }

impl Default for P2pConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bootstrap_peers: Vec::new(),
            listen_port: default_listen_port(),
            run_id: default_run_id(),
            role: default_role(),
            mdns: default_mdns(),
        }
    }
}

// ── Gossip Topics ──────────────────────────────────────────────────

impl P2pConfig {
    /// Gossipsub topic for checkpoint announcements.
    pub fn checkpoint_topic(&self) -> String {
        format!("/distrain/{}/checkpoints", self.run_id)
    }

    /// Gossipsub topic for coordinator-to-coordinator sync.
    pub fn coordinator_sync_topic(&self) -> String {
        format!("/distrain/{}/coordinator_sync", self.run_id)
    }

    /// Gossipsub topic for peer delta exchange (coordinator-optional mode).
    pub fn peer_delta_topic(&self) -> String {
        format!("/distrain/{}/peer_deltas", self.run_id)
    }

    /// DHT key for coordinator registration.
    pub fn coordinator_dht_key(&self) -> String {
        format!("/distrain/coordinator/{}", self.run_id)
    }
}

// ── Gossip Messages ────────────────────────────────────────────────

/// Checkpoint announcement gossiped by the coordinator after producing
/// a new checkpoint. Nodes that receive this start downloading immediately.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointAnnouncement {
    /// New checkpoint version.
    pub version: u64,
    /// R2/S3 path to the checkpoint file.
    pub r2_path: String,
    /// R2/S3 path to the checkpoint delta (from previous version).
    pub delta_path: Option<String>,
    /// Average training loss at this checkpoint.
    pub loss: f64,
    /// Total tokens trained up to this checkpoint.
    pub total_tokens: u64,
    /// When the checkpoint was produced.
    pub timestamp: DateTime<Utc>,
    /// Who produced it (coordinator peer ID or node ID).
    pub produced_by: String,
    /// Number of delta contributions that went into this checkpoint.
    pub num_contributions: u64,
}

/// Coordinator gossip message for multi-coordinator sync.
/// When coordinator B receives coordinator A's checkpoint, it can merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorSyncMessage {
    /// Coordinator's unique identifier.
    pub coordinator_id: String,
    /// Checkpoint version.
    pub checkpoint_version: u64,
    /// R2/S3 path to the checkpoint.
    pub checkpoint_r2_path: String,
    /// Number of contributions in this checkpoint.
    pub num_contributions: u64,
    /// Total tokens processed.
    pub total_tokens: u64,
    /// Average loss.
    pub loss: f64,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

/// Peer delta metadata gossiped between nodes in coordinator-optional mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerDeltaAnnouncement {
    /// Node that produced the delta.
    pub node_id: String,
    /// Checkpoint version the delta was trained against.
    pub checkpoint_version: u64,
    /// R2/S3 key where the delta is stored.
    pub delta_key: String,
    /// Number of inner steps.
    pub inner_steps: u64,
    /// Training loss.
    pub training_loss: f64,
    /// Tokens processed.
    pub tokens_processed: u64,
    /// Weight (tokens × staleness decay).
    pub weight: f64,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

// ── Operating Mode ─────────────────────────────────────────────────

/// The system's operating mode, determined at startup based on
/// available connectivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingMode {
    /// DHT + gossip + multiple coordinators available.
    FullP2p,
    /// DHT + gossip + one coordinator.
    SingleCoordinatorWithDht,
    /// No DHT, direct coordinator URL (current v0.1 behavior).
    DirectHttp,
    /// No coordinator reachable, peers available — nodes merge among themselves.
    PeerMerge,
    /// No coordinator, no peers — train locally, queue deltas.
    Solo,
}

impl std::fmt::Display for OperatingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperatingMode::FullP2p => write!(f, "full_p2p (DHT + gossip + multi-coordinator)"),
            OperatingMode::SingleCoordinatorWithDht => write!(f, "single_coordinator_dht (DHT + gossip)"),
            OperatingMode::DirectHttp => write!(f, "direct_http (no P2P, current behavior)"),
            OperatingMode::PeerMerge => write!(f, "peer_merge (nodes merge, no coordinator)"),
            OperatingMode::Solo => write!(f, "solo (train locally, queue deltas)"),
        }
    }
}
