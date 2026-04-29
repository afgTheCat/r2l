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

/// Control-flow result returned by algorithm hook methods.
///
/// Hook implementations use this to signal whether the surrounding training
/// loop should continue or stop at the current hook boundary.
pub enum HookResult {
    /// Continue the current training loop.
    Continue,
    /// Stop the current training loop at the current hook boundary.
    Break,
}

#[macro_export]
macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            $crate::HookResult::Continue => {}
            $crate::HookResult::Break => return Ok(()),
        }
    };
}
