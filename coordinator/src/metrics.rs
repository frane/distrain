//! Prometheus-compatible metrics for the coordinator.
//!
//! Exports metrics at GET /metrics in Prometheus text exposition format.

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Global metrics singleton.
pub static METRICS: Lazy<Metrics> = Lazy::new(Metrics::default);

pub struct Metrics {
    pub checkpoint_version: AtomicU64,
    pub delta_pushes_total: AtomicU64,
    pub delta_pushes_accepted: AtomicU64,
    pub delta_pushes_rejected: AtomicU64,
    pub aggregations_total: AtomicU64,
    pub aggregations_failed: AtomicU64,
    pub accumulator_contributions: AtomicU64,
    pub active_nodes: AtomicU64,
    pub registered_nodes: AtomicU64,
    pub last_aggregation_secs: Mutex<f64>,
    pub last_outer_delta_norm: Mutex<f64>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            checkpoint_version: AtomicU64::new(0),
            delta_pushes_total: AtomicU64::new(0),
            delta_pushes_accepted: AtomicU64::new(0),
            delta_pushes_rejected: AtomicU64::new(0),
            aggregations_total: AtomicU64::new(0),
            aggregations_failed: AtomicU64::new(0),
            accumulator_contributions: AtomicU64::new(0),
            active_nodes: AtomicU64::new(0),
            registered_nodes: AtomicU64::new(0),
            last_aggregation_secs: Mutex::new(0.0),
            last_outer_delta_norm: Mutex::new(0.0),
        }
    }
}

impl Metrics {
    /// Render metrics in Prometheus text exposition format.
    pub fn render(&self) -> String {
        let agg_secs = *self.last_aggregation_secs.lock().unwrap();
        let delta_norm = *self.last_outer_delta_norm.lock().unwrap();

        format!(
            "\
# HELP distrain_checkpoint_version Current checkpoint version.\n\
# TYPE distrain_checkpoint_version gauge\n\
distrain_checkpoint_version {}\n\
\n\
# HELP distrain_delta_pushes_total Total delta push requests.\n\
# TYPE distrain_delta_pushes_total counter\n\
distrain_delta_pushes_total {}\n\
\n\
# HELP distrain_delta_pushes_accepted Accepted delta pushes.\n\
# TYPE distrain_delta_pushes_accepted counter\n\
distrain_delta_pushes_accepted {}\n\
\n\
# HELP distrain_delta_pushes_rejected Rejected delta pushes.\n\
# TYPE distrain_delta_pushes_rejected counter\n\
distrain_delta_pushes_rejected {}\n\
\n\
# HELP distrain_aggregations_total Total aggregation runs.\n\
# TYPE distrain_aggregations_total counter\n\
distrain_aggregations_total {}\n\
\n\
# HELP distrain_aggregations_failed Failed aggregation runs.\n\
# TYPE distrain_aggregations_failed counter\n\
distrain_aggregations_failed {}\n\
\n\
# HELP distrain_accumulator_contributions Current contributions in accumulator.\n\
# TYPE distrain_accumulator_contributions gauge\n\
distrain_accumulator_contributions {}\n\
\n\
# HELP distrain_active_nodes Number of active nodes.\n\
# TYPE distrain_active_nodes gauge\n\
distrain_active_nodes {}\n\
\n\
# HELP distrain_registered_nodes Total registered nodes.\n\
# TYPE distrain_registered_nodes gauge\n\
distrain_registered_nodes {}\n\
\n\
# HELP distrain_last_aggregation_seconds Duration of last aggregation.\n\
# TYPE distrain_last_aggregation_seconds gauge\n\
distrain_last_aggregation_seconds {:.4}\n\
\n\
# HELP distrain_last_outer_delta_norm Delta norm of last aggregation.\n\
# TYPE distrain_last_outer_delta_norm gauge\n\
distrain_last_outer_delta_norm {:.6}\n",
            self.checkpoint_version.load(Ordering::Relaxed),
            self.delta_pushes_total.load(Ordering::Relaxed),
            self.delta_pushes_accepted.load(Ordering::Relaxed),
            self.delta_pushes_rejected.load(Ordering::Relaxed),
            self.aggregations_total.load(Ordering::Relaxed),
            self.aggregations_failed.load(Ordering::Relaxed),
            self.accumulator_contributions.load(Ordering::Relaxed),
            self.active_nodes.load(Ordering::Relaxed),
            self.registered_nodes.load(Ordering::Relaxed),
            agg_secs,
            delta_norm,
        )
    }
}
