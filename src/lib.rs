//! A batched spiking language model implemented with Burn tensors and autodiff.
//!
//! Model dynamics and the explicitly chosen surrogate learning rule are derived
//! in `docs/architecture.md`. WGPU training and Flex CPU tests execute the same
//! model; dataset import and all project utilities are Rust.

pub mod checkpoint;
pub mod data;
pub mod fineweb;
pub mod generation_probe;
pub mod language_probe;
pub mod memory_probe;
pub mod model;
pub mod probe;
pub mod runtime;
pub mod text_probe;
pub mod training;
pub mod understanding_probe;

mod continuation_score;
mod rng;
mod text_activity;
mod understanding_cases;
