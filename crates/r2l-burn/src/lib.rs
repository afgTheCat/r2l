//! Burn-backed policy and learner implementations for `r2l`.
//!
//! This crate provides the Burn implementations used by the higher-level
//! on-policy APIs in the workspace. Its public surface is centered on:
//! - [`distributions`], which contains Burn policy implementations for
//!   discrete and Box action spaces
//! - [`learning_module`], which contains Burn
//!   [`OnPolicyLearner`](r2l_core::on_policy::learning_module::OnPolicyLearner)
//!   implementations for policy/value training
//!
//! Most users interact with these types indirectly through `r2l`, but they
//! remain public for lower-level composition and backend-specific work.

/// Burn policy implementations for supported action spaces.
pub mod distributions;
/// Burn policy/value learners and associated loss types.
pub mod learning_module;
mod sequential;
