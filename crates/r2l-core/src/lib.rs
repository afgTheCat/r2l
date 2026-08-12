//! Core traits and data types shared by the `r2l` workspace.
//!
//! `r2l-core` is the contracts crate. It defines the small set of interfaces
//! that environments, samplers, policies, agents, learners, and tensor
//! backends agree on. Backend-specific implementations live in crates such as
//! `r2l-burn` and `r2l-candle`; concrete algorithms and builders live outside
//! this crate as well.
//!
//! Most downstream code should start with the prelude:
//!
//! ```
//! use r2l_core::prelude::*;
//! ```
//!
//! The main extension points are:
//!
//! - [`Env`] and [`EnvBuilder`] for environment integrations.
//! - [`R2lTensor`] for tensor types used by environments
//!   and learning code.
//! - [`Actor`], [`Policy`], [`ValueFunction`], and [`Learner`] for model
//!   and optimizer components.
//! - [`TrajectoryBuffer`] and [`TrajectoryView`] for rollout storage.
//! - [`Agent`], [`Sampler`], and [`OnPolicyAlgorithm`] for on-policy training
//!   loops.
//!
//! [`Actor`]: crate::models::Actor
//! [`Agent`]: crate::on_policy::algorithm::Agent
//! [`Env`]: crate::env::Env
//! [`EnvBuilder`]: crate::env::EnvBuilder
//! [`Learner`]: crate::models::Learner
//! [`OnPolicyAlgorithm`]: crate::on_policy::algorithm::OnPolicyAlgorithm
//! [`Policy`]: crate::models::Policy
//! [`R2lTensor`]: crate::tensor::R2lTensor
//! [`Sampler`]: crate::on_policy::algorithm::Sampler
//! [`TrajectoryBuffer`]: crate::buffers::buffer::TrajectoryBuffer
//! [`TrajectoryView`]: crate::buffers::buffer::TrajectoryView
//! [`ValueFunction`]: crate::models::ValueFunction

/// Rollout transition and trajectory storage.
pub mod buffers;
/// Environment traits and space descriptions.
pub mod env;
/// Error types
pub mod error;
/// Actor, policy, value-function, and learner traits.
pub mod models;
/// Shared interfaces for on-policy training loops.
pub mod on_policy;
/// Reproducible random-number utilities.
pub mod rng;
/// Online mean and variance estimators.
pub mod running_mean;
/// Backend-neutral tensor interfaces and adapters.
pub mod tensor;
mod utils;

pub use utils::actor_wrapper::ActorWrapper;

/// Control-flow result returned by training hooks.
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
/// Breaks out of the surrounding loop when a hook requests [`HookResult::Break`].
macro_rules! break_on_hook_result {
    ($hook_res:expr) => {
        match $hook_res {
            $crate::HookResult::Continue => {}
            $crate::HookResult::Break => break,
        }
    };
}

#[macro_export]
/// Returns `Ok(())` from the surrounding function when a hook requests
/// [`HookResult::Break`].
macro_rules! return_on_hook_result {
    ($hook_res:expr) => {
        match $hook_res {
            $crate::HookResult::Continue => {}
            $crate::HookResult::Break => return Ok(()),
        }
    };
}

/// Common imports for implementing environments, policies, agents, samplers,
/// and learners.
pub mod prelude {
    pub use crate::HookResult;
    pub use crate::buffers::Memory;
    pub use crate::env::{Env, EnvBuilder, EnvBuilderType, EnvDescription, Space};
    pub use crate::models::{
        ActivationFunction, Actor, Learner, Policy, ToSafetensors, ValueFunction,
    };
    pub use crate::on_policy::learning_module::OnPolicyLearner;
    pub use crate::on_policy::losses::FromPolicyValueLosses;
    pub use crate::tensor::{R2lTensor, TensorData};
}
