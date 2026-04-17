#![recursion_limit = "256"]
//! Distrain node library — reusable training logic for CLI, desktop, etc.

pub mod client;
pub mod continuous;
pub mod data;
pub mod peer_merge;
pub mod resources;
pub mod resume;
pub mod trainer;

#[cfg(test)]
mod data_tests;
#[cfg(test)]
mod trainer_tests;
