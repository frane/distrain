//! Fallback cascade: determine operating mode from available connectivity.
//!
//! The system degrades gracefully through 5 levels:
//!
//! 1. FullP2p — DHT + gossip + multiple coordinators
//! 2. SingleCoordinatorWithDht — DHT + gossip + one coordinator
//! 3. DirectHttp — no DHT, direct coordinator URL (current v0.1 behavior)
//! 4. PeerMerge — no coordinator, peers available (nodes merge among themselves)
//! 5. Solo — no coordinator, no peers (train locally, queue deltas)
//!
//! The system works at level 3 with zero code changes. Levels 1-2 and 4-5
//! are additive.

use tracing::info;

use super::types::{OperatingMode, P2pConfig};

/// Determine the initial operating mode based on configuration and connectivity.
///
/// Called at startup. The mode can be upgraded later if connectivity improves
/// (e.g., coordinator comes online while in peer merge mode).
pub fn determine_operating_mode(
    p2p_config: &P2pConfig,
    coordinator_url: &str,
    p2p_available: bool,
    coordinator_reachable: bool,
    num_dht_coordinators: usize,
    num_peers: usize,
) -> OperatingMode {
    let mode = if !p2p_config.enabled {
        // P2P disabled — direct HTTP only
        if coordinator_reachable {
            OperatingMode::DirectHttp
        } else {
            OperatingMode::Solo
        }
    } else if p2p_available {
        if num_dht_coordinators > 1 {
            OperatingMode::FullP2p
        } else if num_dht_coordinators == 1 || coordinator_reachable {
            OperatingMode::SingleCoordinatorWithDht
        } else if num_peers > 0 {
            OperatingMode::PeerMerge
        } else {
            OperatingMode::Solo
        }
    } else {
        // P2P enabled but service failed to start
        if coordinator_reachable {
            OperatingMode::DirectHttp
        } else {
            OperatingMode::Solo
        }
    };

    info!("Operating mode: {mode}");
    if mode == OperatingMode::Solo {
        info!("Solo mode: will train locally and queue deltas until connectivity returns");
    }
    if mode == OperatingMode::PeerMerge {
        info!("Peer merge mode: nodes will merge deltas among themselves (no coordinator)");
    }

    mode
}

/// Check if the mode should be upgraded based on new connectivity info.
///
/// Called periodically or when connectivity changes (coordinator comes online,
/// peers discovered, etc). Returns the new mode if different from current.
pub fn maybe_upgrade_mode(
    current: OperatingMode,
    coordinator_reachable: bool,
    num_dht_coordinators: usize,
    num_peers: usize,
    p2p_available: bool,
) -> Option<OperatingMode> {
    let new_mode = if p2p_available && num_dht_coordinators > 1 {
        OperatingMode::FullP2p
    } else if p2p_available && (num_dht_coordinators == 1 || coordinator_reachable) {
        OperatingMode::SingleCoordinatorWithDht
    } else if coordinator_reachable {
        OperatingMode::DirectHttp
    } else if p2p_available && num_peers > 0 {
        OperatingMode::PeerMerge
    } else {
        OperatingMode::Solo
    };

    if new_mode != current {
        info!("Mode upgrade: {current} → {new_mode}");
        Some(new_mode)
    } else {
        None
    }
}
