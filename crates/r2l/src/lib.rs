//! r2l is a reinforcement learning library written in rust. The goal of r2l is two-fold:
//!
//! - To provide a convinient high level builder API with a feature set similar to sb3
//! - To have all the lower level builder types and triats be exposed
//!
//! In fact, the hihg level API is just an implementation of the lower level components --
//! it serves as a testament to what is possible with the exposed pieces.
//!
//!
//! ## Hello r2l
//!
//! Possibly one of the simplest hello world style app is:
//!
//! ```no_run
//! use r2l::{LearningSchedule, PPOAlgorithmBuilder};
//!
//! let mut algorithm = PPOAlgorithmBuilder::gym("Pendulum-v1", 4).build().unwrap();
//! algorithm.train().unwrap();
//! ```
//!
//! While this example is small, using the [`PPOAlgorithmBuilder`], it is also highly
//! customizable. Check the relevant page for [`PPOAlgorithmBuilder`]. If you wish to
//! start out with A2C, check out [`A2CAlgorithmBuilder`].
//!
//! ## What is covered here
//!
//! The current focus of r2l is on-policy algorithms. This release, we have `PPO` and
//! `A2C` implemented, supporting the more simple policy types:
//!
//! - Diag gaussian
//! - Categorical
//! -

#![warn(missing_docs)]

use burn::backend::{Autodiff, NdArray};

// builders + hooks + higher level helpers
mod builders;
mod evaluators;
mod hooks;
mod utils;

/// Default autodifferentiation backend used by Burn-based builders.
pub type BurnBackend = Autodiff<NdArray>;

pub use builders::{
    A2CAlgorithmBuilder, A2CBurn, A2CCandle, AdamWParams, InferenceArtifacts, InferenceRunner,
    OnPolicyAlgoBuilder, OnPolicyOptimizerLayout, PPOAlgorithmBuilder, PPOBurn, PPOCandle,
    PolicyBuilder, TrainingArtifactsConfig,
};
pub use evaluators::best_actor_evaluator::EvaluationSettings;
pub use hooks::a2c::{A2CMinibatchStats, A2CRolloutStats};
pub use hooks::on_policy::{
    DefaultOnPolicyAlgorithmHooks, LearningRateSchedule, LearningSchedule, OnPolicyCommand,
    OnPolicyCommandReceiver, OnPolicyCommandResult, OnPolicyCommandSender,
    on_policy_command_channel,
};
pub use hooks::ppo::{PPOMinibatchStats, PPORolloutStats};
pub use hooks::sampler::{EpisodeBoundHook, StepBoundHook};
pub use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    models::ActivationFunction,
    on_policy::algorithm::OnPolicyAlgorithm,
    tensor::TensorData,
};
pub use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};
