//! Shared contracts for on-policy algorithms.

/// Agent, sampler, adapter, and training-loop interfaces.
pub mod algorithm;
/// Learning-module interface used by on-policy agents.
pub mod learning_module;
/// Policy/value loss containers.
pub mod losses;
