//! Importance-weighted parameter selection.
//!
//! Tracks per-parameter running average of recent movement (|delta|).
//! High importance = large delta relative to historical movement.
//! Used for top-k/block selection instead of raw magnitude.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Tracks per-parameter importance via EMA of |delta|.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportanceTracker {
    /// EMA of |delta| per parameter (flattened), keyed by param name.
    pub running_movement: HashMap<String, Vec<f32>>,
    /// EMA decay factor. Default 0.9 (recent rounds matter more).
    pub alpha: f32,
}

impl ImportanceTracker {
    pub fn new(alpha: f32) -> Self {
        Self {
            running_movement: HashMap::new(),
            alpha,
        }
    }

    /// Compute importance scores for a delta.
    /// importance = |delta| / (running_movement + eps)
    /// High values = params that moved more than usual (novel signal).
    pub fn compute_importance(
        &self,
        delta: &HashMap<String, Vec<f32>>,
    ) -> HashMap<String, Vec<f32>> {
        let eps = 1e-8f32;
        let mut importance = HashMap::new();

        for (name, values) in delta {
            let imp = if let Some(movement) = self.running_movement.get(name) {
                values
                    .iter()
                    .zip(movement.iter())
                    .map(|(d, m)| d.abs() / (m + eps))
                    .collect()
            } else {
                // No history: all params equally important (importance = |delta|)
                values.iter().map(|d| d.abs()).collect()
            };
            importance.insert(name.clone(), imp);
        }

        importance
    }

    /// Compute per-row importance for block sparsity.
    /// Returns mean importance per row for 2D tensors.
    pub fn compute_row_importance(
        &self,
        delta: &HashMap<String, Vec<f32>>,
        shapes: &HashMap<String, Vec<usize>>,
    ) -> HashMap<String, Vec<f32>> {
        let element_importance = self.compute_importance(delta);
        let mut row_importance = HashMap::new();

        for (name, imp) in &element_importance {
            let shape = shapes.get(name);
            if let Some(s) = shape {
                if s.len() == 2 {
                    let rows = s[0];
                    let cols = s[1];
                    let mut row_imp = Vec::with_capacity(rows);
                    for r in 0..rows {
                        let start = r * cols;
                        let end = (start + cols).min(imp.len());
                        let mean: f32 = imp[start..end].iter().sum::<f32>() / cols as f32;
                        row_imp.push(mean);
                    }
                    row_importance.insert(name.clone(), row_imp);
                    continue;
                }
            }
            // 1D or no shape: importance is per-element
            row_importance.insert(name.clone(), imp.clone());
        }

        row_importance
    }

    /// Update the running movement EMA with a new delta.
    /// Call after each training round.
    pub fn update(&mut self, delta: &HashMap<String, Vec<f32>>) {
        let alpha = self.alpha;
        for (name, values) in delta {
            let entry = self
                .running_movement
                .entry(name.clone())
                .or_insert_with(|| vec![0.0f32; values.len()]);

            // Resize if needed (model parameter count changed — shouldn't happen,
            // but handle gracefully)
            if entry.len() != values.len() {
                *entry = vec![0.0f32; values.len()];
            }

            for (m, d) in entry.iter_mut().zip(values.iter()) {
                *m = alpha * *m + (1.0 - alpha) * d.abs();
            }
        }
    }

    /// Number of tracked tensors.
    pub fn num_tensors(&self) -> usize {
        self.running_movement.len()
    }

    /// Total number of tracked parameters.
    pub fn num_params(&self) -> usize {
        self.running_movement.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_no_history() {
        let tracker = ImportanceTracker::new(0.9);
        let mut delta = HashMap::new();
        delta.insert("w".to_string(), vec![1.0, -2.0, 0.5]);

        let imp = tracker.compute_importance(&delta);
        // No history: importance ≈ |delta| (with eps denominator)
        assert!((imp["w"][0] - 1.0).abs() < 0.01);
        assert!((imp["w"][1] - 2.0).abs() < 0.01);
        assert!((imp["w"][2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_importance_with_history() {
        let mut tracker = ImportanceTracker::new(0.9);

        // Feed 10 rounds with constant movement
        for _ in 0..10 {
            let mut delta = HashMap::new();
            delta.insert("w".to_string(), vec![1.0, 1.0, 1.0]);
            tracker.update(&delta);
        }

        // Now a delta where param 0 barely moved, param 2 moved a lot
        let mut test_delta = HashMap::new();
        test_delta.insert("w".to_string(), vec![0.1, 1.0, 5.0]);
        let imp = tracker.compute_importance(&test_delta);

        // param 0: small delta / ~1.0 history = low importance
        // param 2: large delta / ~1.0 history = high importance
        assert!(imp["w"][2] > imp["w"][0], "High-movement param should have higher importance");
        assert!(imp["w"][2] > imp["w"][1], "5x movement should outrank 1x");
    }

    #[test]
    fn test_row_importance_2d() {
        let tracker = ImportanceTracker::new(0.9);
        let mut delta = HashMap::new();
        // 3 rows × 2 cols, row 1 has highest values
        delta.insert("w".to_string(), vec![0.1, 0.1, 5.0, 5.0, 0.5, 0.5]);
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), vec![3, 2]);

        let row_imp = tracker.compute_row_importance(&delta, &shapes);
        assert_eq!(row_imp["w"].len(), 3);
        // Row 1 should have highest importance
        assert!(row_imp["w"][1] > row_imp["w"][0]);
        assert!(row_imp["w"][1] > row_imp["w"][2]);
    }

    #[test]
    fn test_update_ema() {
        let mut tracker = ImportanceTracker::new(0.5);

        let mut d1 = HashMap::new();
        d1.insert("w".to_string(), vec![2.0]);
        tracker.update(&d1);
        // After first update: movement = 0.5 * 0 + 0.5 * 2.0 = 1.0
        assert!((tracker.running_movement["w"][0] - 1.0).abs() < 1e-6);

        let mut d2 = HashMap::new();
        d2.insert("w".to_string(), vec![4.0]);
        tracker.update(&d2);
        // After second update: movement = 0.5 * 1.0 + 0.5 * 4.0 = 2.5
        assert!((tracker.running_movement["w"][0] - 2.5).abs() < 1e-6);
    }
}
