//! Candle-backed policy and learner implementations for `r2l`.
//!
//! This crate provides the Candle implementations used by the higher-level
//! on-policy APIs in the workspace. Its public surface is centered on:
//! - [`distributions`], which contains Candle policy implementations for
//!   discrete and Box action spaces
//! - [`learning_module`], which contains a Candle
//!   [`OnPolicyLearner`](r2l_core::on_policy::learning_module::OnPolicyLearner)
//!   implementation for policy/value training
//!
//! Most users interact with these types indirectly through `r2l`, but they
//! remain public for lower-level composition.

/// Candle policy implementations for supported action spaces.
pub mod distributions;
/// Candle policy/value learners and associated loss types.
pub mod learning_module;
/// Creates a variable builder that initializes new variables with r2l's seeded RNG.
pub use random::seeded_var_builder;

mod optimizer;
mod random;
mod sequential;
