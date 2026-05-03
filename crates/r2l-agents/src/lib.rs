//! Core RL algorithm implementations used by higher-level `r2l` crates.
//!
//! This crate contains lower-level on-policy learning algorithms such as A2C,
//! PPO, and VPG together with their hook interfaces and shared rollout
//! processing utilities.
//!
//! Most users interact with these algorithms through `r2l-api`, which provides
//! builders, backend selection, and default hooks on top of this crate.

/// On-policy algorithm implementations and shared rollout-processing helpers.
pub mod on_policy_algorithms;

use r2l_core::HookResult;
