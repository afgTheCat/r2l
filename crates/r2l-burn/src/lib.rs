//! Burn-backed policy and learning-module implementations for `r2l`.
//!
//! This crate provides the Burn implementations used by the higher-level
//! on-policy APIs in the workspace. Its public surface is centered on:
//! - [`distributions`], which contains Burn policy implementations for
//!   discrete and Box action spaces
//! - [`learning_module`], which contains Burn
//!   [`OnPolicyLearningModule`](r2l_core::on_policy::learning_module::OnPolicyLearningModule)
//!   implementations for policy/value training
//!
//! Most users interact with these types indirectly through `r2l-api`, but they
//! remain public for lower-level composition and backend-specific work.

/// Burn policy implementations for supported action spaces.
pub mod distributions;
/// Burn policy/value learning modules and associated loss types.
pub mod learning_module;
mod sequential;
