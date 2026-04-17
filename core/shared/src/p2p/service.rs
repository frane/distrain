//! P2P discovery service: libp2p swarm with Kademlia DHT + gossipsub.
//!
//! Provides:
//! - Coordinator registration on DHT (coordinators announce themselves)
//! - Coordinator lookup via DHT (nodes find coordinators)
//! - Checkpoint announcements via gossipsub (push, not poll)
//! - Coordinator-to-coordinator sync via gossipsub
//! - Peer delta exchange via gossipsub (coordinator-optional mode)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use anyhow::{Context, Result};
use libp2p::futures::StreamExt;
use libp2p::gossipsub::{self, IdentTopic, MessageAuthenticity};
use libp2p::identity::Keypair;
use libp2p::kad::{self, store::MemoryStore};
use libp2p::swarm::SwarmEvent;
use libp2p::{Multiaddr, PeerId, SwarmBuilder};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::types::*;

/// Events emitted by the P2P service to the application layer.
#[derive(Debug, Clone)]
pub enum P2pEvent {
    /// A new checkpoint was announced via gossip.
    CheckpointAnnounced(CheckpointAnnouncement),
    /// A coordinator was discovered via DHT.
    CoordinatorDiscovered {
        peer_id: String,
        addresses: Vec<String>,
    },
    /// A coordinator sync message was received.
    CoordinatorSync(CoordinatorSyncMessage),
    /// A peer delta was announced (coordinator-optional mode).
    PeerDelta(PeerDeltaAnnouncement),
    /// A new peer connected.
    PeerConnected(String),
    /// A peer disconnected.
    PeerDisconnected(String),
}

/// Handle to the running P2P service. Used to send commands.
#[derive(Clone)]
pub struct P2pHandle {
    cmd_tx: mpsc::Sender<P2pCommand>,
    pub local_peer_id: String,
}

/// Commands sent to the P2P service from the application layer.
enum P2pCommand {
    /// Publish a checkpoint announcement via gossipsub.
    AnnounceCheckpoint(CheckpointAnnouncement),
    /// Publish a coordinator sync message.
    SyncCoordinator(CoordinatorSyncMessage),
    /// Publish a peer delta announcement.
    AnnouncePeerDelta(PeerDeltaAnnouncement),
    /// Register as coordinator on DHT.
    RegisterCoordinator(String), // HTTP address (e.g., "http://1.2.3.4:8000")
    /// Look up coordinator addresses from DHT.
    LookupCoordinator,
}

impl P2pHandle {
    /// Announce a new checkpoint to all peers.
    pub async fn announce_checkpoint(&self, announcement: CheckpointAnnouncement) -> Result<()> {
        self.cmd_tx
            .send(P2pCommand::AnnounceCheckpoint(announcement))
            .await
            .map_err(|_| anyhow::anyhow!("P2P service shut down"))
    }

    /// Send a coordinator sync message to other coordinators.
    pub async fn sync_coordinator(&self, msg: CoordinatorSyncMessage) -> Result<()> {
        self.cmd_tx
            .send(P2pCommand::SyncCoordinator(msg))
            .await
            .map_err(|_| anyhow::anyhow!("P2P service shut down"))
    }

    /// Announce a peer delta (coordinator-optional mode).
    pub async fn announce_peer_delta(&self, msg: PeerDeltaAnnouncement) -> Result<()> {
        self.cmd_tx
            .send(P2pCommand::AnnouncePeerDelta(msg))
            .await
            .map_err(|_| anyhow::anyhow!("P2P service shut down"))
    }

    /// Register this node as a coordinator on DHT.
    pub async fn register_coordinator(&self, http_address: String) -> Result<()> {
        self.cmd_tx
            .send(P2pCommand::RegisterCoordinator(http_address))
            .await
            .map_err(|_| anyhow::anyhow!("P2P service shut down"))
    }

    /// Trigger a coordinator lookup via DHT.
    pub async fn lookup_coordinator(&self) -> Result<()> {
        self.cmd_tx
            .send(P2pCommand::LookupCoordinator)
            .await
            .map_err(|_| anyhow::anyhow!("P2P service shut down"))
    }
}

/// Combined libp2p behaviour: Kademlia DHT + Gossipsub + Identify.
#[derive(libp2p::swarm::NetworkBehaviour)]
struct DistrainBehaviour {
    kademlia: kad::Behaviour<MemoryStore>,
    gossipsub: gossipsub::Behaviour,
    identify: libp2p::identify::Behaviour,
}

/// Start the P2P service. Returns a handle for commands and a receiver for events.
///
/// The service runs as a background tokio task. It will:
/// 1. Create a libp2p swarm with Kademlia + Gossipsub
/// 2. Listen on the configured port
/// 3. Connect to bootstrap peers
/// 4. Subscribe to gossip topics
/// 5. Process commands from the handle
/// 6. Emit events when gossip messages arrive or peers are discovered
pub async fn start_p2p_service(
    config: &P2pConfig,
) -> Result<(P2pHandle, mpsc::Receiver<P2pEvent>)> {
    let local_key = Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    info!("P2P: local peer ID = {local_peer_id}");

    // Kademlia DHT
    let store = MemoryStore::new(local_peer_id);
    let kademlia = kad::Behaviour::new(local_peer_id, store);

    // Gossipsub
    let message_id_fn = |message: &gossipsub::Message| {
        let mut hasher = DefaultHasher::new();
        message.data.hash(&mut hasher);
        message.topic.hash(&mut hasher);
        gossipsub::MessageId::from(hasher.finish().to_string())
    };
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(10))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .message_id_fn(message_id_fn)
        .build()
        .context("Failed to build gossipsub config")?;
    let gossipsub = gossipsub::Behaviour::new(
        MessageAuthenticity::Signed(local_key.clone()),
        gossipsub_config,
    )
    .map_err(|e| anyhow::anyhow!("Failed to create gossipsub behaviour: {e}"))?;

    // Identify protocol (required for Kademlia to work properly)
    let identify = libp2p::identify::Behaviour::new(libp2p::identify::Config::new(
        "/distrain/0.2.0".to_string(),
        local_key.public(),
    ));

    let behaviour = DistrainBehaviour {
        kademlia,
        gossipsub,
        identify,
    };

    let mut swarm = SwarmBuilder::with_existing_identity(local_key)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )
        .context("Failed to configure TCP transport")?
        .with_behaviour(|_| Ok(behaviour))
        .context("Failed to build swarm behaviour")?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Listen on configured port
    let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", config.listen_port)
        .parse()
        .context("Failed to parse listen address")?;
    swarm.listen_on(listen_addr)?;

    // Subscribe to gossip topics
    let checkpoint_topic = IdentTopic::new(config.checkpoint_topic());
    let coordinator_sync_topic = IdentTopic::new(config.coordinator_sync_topic());
    let peer_delta_topic = IdentTopic::new(config.peer_delta_topic());
    swarm.behaviour_mut().gossipsub.subscribe(&checkpoint_topic)?;
    swarm.behaviour_mut().gossipsub.subscribe(&coordinator_sync_topic)?;
    swarm.behaviour_mut().gossipsub.subscribe(&peer_delta_topic)?;
    info!(
        "P2P: subscribed to topics: {}, {}, {}",
        config.checkpoint_topic(),
        config.coordinator_sync_topic(),
        config.peer_delta_topic(),
    );

    // Connect to bootstrap peers
    for addr_str in &config.bootstrap_peers {
        match addr_str.parse::<Multiaddr>() {
            Ok(addr) => {
                // Extract peer ID from multiaddr if present
                if let Some(libp2p::multiaddr::Protocol::P2p(peer_id)) = addr.iter().last() {
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                    info!("P2P: added bootstrap peer {peer_id} at {addr}");
                } else {
                    warn!("P2P: bootstrap address missing peer ID: {addr_str}");
                }
            }
            Err(e) => warn!("P2P: invalid bootstrap address {addr_str}: {e}"),
        }
    }

    // Bootstrap Kademlia
    if !config.bootstrap_peers.is_empty() {
        if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
            warn!("P2P: Kademlia bootstrap failed: {e}");
        }
    }

    let (cmd_tx, mut cmd_rx) = mpsc::channel::<P2pCommand>(64);
    let (event_tx, event_rx) = mpsc::channel::<P2pEvent>(256);

    let handle = P2pHandle {
        cmd_tx,
        local_peer_id: local_peer_id.to_string(),
    };

    let dht_key = config.coordinator_dht_key();
    let ckpt_topic_str = config.checkpoint_topic();
    let sync_topic_str = config.coordinator_sync_topic();
    let delta_topic_str = config.peer_delta_topic();

    // Spawn the swarm event loop
    tokio::spawn(async move {
        loop {
            tokio::select! {
                // Process swarm events
                event = swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(DistrainBehaviourEvent::Gossipsub(
                            gossipsub::Event::Message { message, .. },
                        )) => {
                            let topic = message.topic.as_str();
                            if topic == ckpt_topic_str {
                                if let Ok(ann) = serde_json::from_slice::<CheckpointAnnouncement>(&message.data) {
                                    debug!("P2P: received checkpoint announcement v{}", ann.version);
                                    let _ = event_tx.send(P2pEvent::CheckpointAnnounced(ann)).await;
                                }
                            } else if topic == sync_topic_str {
                                if let Ok(msg) = serde_json::from_slice::<CoordinatorSyncMessage>(&message.data) {
                                    debug!("P2P: received coordinator sync from {}", msg.coordinator_id);
                                    let _ = event_tx.send(P2pEvent::CoordinatorSync(msg)).await;
                                }
                            } else if topic == delta_topic_str {
                                if let Ok(msg) = serde_json::from_slice::<PeerDeltaAnnouncement>(&message.data) {
                                    debug!("P2P: received peer delta from {}", msg.node_id);
                                    let _ = event_tx.send(P2pEvent::PeerDelta(msg)).await;
                                }
                            }
                        }
                        SwarmEvent::Behaviour(DistrainBehaviourEvent::Kademlia(
                            kad::Event::OutboundQueryProgressed { result, .. },
                        )) => {
                            match result {
                                kad::QueryResult::GetRecord(Ok(
                                    kad::GetRecordOk::FoundRecord(kad::PeerRecord { record, .. }),
                                )) => {
                                    if let Ok(value) = String::from_utf8(record.value) {
                                        // Parse coordinator addresses from DHT record
                                        if let Ok(addrs) = serde_json::from_str::<Vec<String>>(&value) {
                                            let peer = record.publisher.map(|p| p.to_string()).unwrap_or_default();
                                            info!("P2P: discovered coordinator via DHT: {addrs:?}");
                                            let _ = event_tx.send(P2pEvent::CoordinatorDiscovered {
                                                peer_id: peer,
                                                addresses: addrs,
                                            }).await;
                                        }
                                    }
                                }
                                kad::QueryResult::GetRecord(Err(e)) => {
                                    debug!("P2P: DHT lookup failed: {e:?}");
                                }
                                kad::QueryResult::PutRecord(Ok(_)) => {
                                    debug!("P2P: DHT record published successfully");
                                }
                                _ => {}
                            }
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            debug!("P2P: connected to {peer_id}");
                            let _ = event_tx.send(P2pEvent::PeerConnected(peer_id.to_string())).await;
                        }
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            debug!("P2P: disconnected from {peer_id}");
                            let _ = event_tx.send(P2pEvent::PeerDisconnected(peer_id.to_string())).await;
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            info!("P2P: listening on {address}/p2p/{local_peer_id}");
                        }
                        _ => {}
                    }
                }

                // Process commands from application
                Some(cmd) = cmd_rx.recv() => {
                    match cmd {
                        P2pCommand::AnnounceCheckpoint(ann) => {
                            let topic = IdentTopic::new(&ckpt_topic_str);
                            if let Ok(data) = serde_json::to_vec(&ann) {
                                match swarm.behaviour_mut().gossipsub.publish(topic, data) {
                                    Ok(_) => debug!("P2P: published checkpoint v{}", ann.version),
                                    Err(e) => warn!("P2P: failed to publish checkpoint: {e}"),
                                }
                            }
                        }
                        P2pCommand::SyncCoordinator(msg) => {
                            let topic = IdentTopic::new(&sync_topic_str);
                            if let Ok(data) = serde_json::to_vec(&msg) {
                                match swarm.behaviour_mut().gossipsub.publish(topic, data) {
                                    Ok(_) => debug!("P2P: published coordinator sync"),
                                    Err(e) => warn!("P2P: failed to publish coordinator sync: {e}"),
                                }
                            }
                        }
                        P2pCommand::AnnouncePeerDelta(msg) => {
                            let topic = IdentTopic::new(&delta_topic_str);
                            if let Ok(data) = serde_json::to_vec(&msg) {
                                match swarm.behaviour_mut().gossipsub.publish(topic, data) {
                                    Ok(_) => debug!("P2P: published peer delta"),
                                    Err(e) => warn!("P2P: failed to publish peer delta: {e}"),
                                }
                            }
                        }
                        P2pCommand::RegisterCoordinator(http_address) => {
                            let key = kad::RecordKey::new(&dht_key);
                            let value = serde_json::to_vec(&vec![http_address.clone()]).unwrap_or_default();
                            let record = kad::Record {
                                key,
                                value,
                                publisher: None,
                                expires: None,
                            };
                            match swarm.behaviour_mut().kademlia.put_record(record, kad::Quorum::One) {
                                Ok(_) => info!("P2P: registered coordinator at {http_address} on DHT"),
                                Err(e) => warn!("P2P: failed to register on DHT: {e}"),
                            }
                        }
                        P2pCommand::LookupCoordinator => {
                            let key = kad::RecordKey::new(&dht_key);
                            swarm.behaviour_mut().kademlia.get_record(key);
                            debug!("P2P: initiated DHT lookup for coordinator");
                        }
                    }
                }
            }
        }
    });

    Ok((handle, event_rx))
}
