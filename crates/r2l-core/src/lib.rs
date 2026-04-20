//! Core traits and data types shared by the `r2l` workspace.
//!
//! `r2l-core` is the contracts crate. It defines the small set of interfaces
//! that environments, samplers, policies, agents, learning modules, and tensor
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
//! - [`Env`] and [`EnvBuilderTrait`] for environment integrations.
//! - [`R2lTensor`] and [`R2lTensorMath`] for tensor types used by environments
//!   and learning code.
//! - [`Actor`], [`Policy`], [`ValueFunction`], and [`LearningModule`] for model
//!   and optimizer components.
//! - [`TrajectoryContainer`] and [`ExpandableTrajectoryContainer`] for rollout
//!   storage.
//! - [`Agent`], [`Sampler`], and [`OnPolicyAlgorithm`] for on-policy training
//!   loops.
//!
//! [`Actor`]: crate::models::Actor
//! [`Agent`]: crate::on_policy::algorithm::Agent
//! [`Env`]: crate::env::Env
//! [`EnvBuilderTrait`]: crate::env::EnvBuilderTrait
//! [`ExpandableTrajectoryContainer`]: crate::buffers::ExpandableTrajectoryContainer
//! [`LearningModule`]: crate::models::LearningModule
//! [`OnPolicyAlgorithm`]: crate::on_policy::algorithm::OnPolicyAlgorithm
//! [`Policy`]: crate::models::Policy
//! [`R2lTensor`]: crate::tensor::R2lTensor
//! [`R2lTensorMath`]: crate::tensor::R2lTensorMath
//! [`Sampler`]: crate::on_policy::algorithm::Sampler
//! [`TrajectoryContainer`]: crate::buffers::TrajectoryContainer
//! [`ValueFunction`]: crate::models::ValueFunction

pub mod buffers;
pub mod env;
pub mod error;
pub mod models;
pub mod on_policy;
pub mod rng;
pub mod tensor;
mod utils;

/// Common imports for implementing environments, policies, agents, samplers,
/// and learning modules.
pub mod prelude {
    pub use crate::buffers::{
        EditableTrajectoryContainer, ExpandableTrajectoryContainer, Memory, TrajectoryContainer,
        fix_sized::FixedSizeStateBuffer, variable_sized::VariableSizedStateBuffer,
    };
    pub use crate::env::{Env, EnvBuilder, EnvBuilderType, EnvDescription, Space};
    pub use crate::models::{Actor, LearningModule, Policy, ValueFunction};
    pub use crate::on_policy::algorithm::{
        Agent, OnPolicyAlgorithm, OnPolicyAlgorithmHooks, Sampler,
    };
    pub use crate::on_policy::learning_module::OnPolicyLearningModule;
    pub use crate::on_policy::losses::FromPolicyValueLosses;
    pub use crate::tensor::{R2lTensor, R2lTensorMath, TensorData};
}
