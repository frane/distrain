//! Peer-to-peer networking: DHT discovery + gossipsub announcements.
//!
//! Provides coordinator discovery (replace hardcoded URLs) and checkpoint
//! propagation (replace polling) via libp2p.
//!
//! Gated behind the `p2p` feature flag. When disabled, the coordinator
//! and node use direct HTTP (current behavior).

pub mod cascade;
#[cfg(feature = "p2p")]
pub mod service;
pub mod types;
