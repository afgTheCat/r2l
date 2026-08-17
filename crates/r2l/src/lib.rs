//! r2l is a reinforcement learning library for Rust. It provides high-level PPO and A2C
//! training builders while exposing lower-level components for customization.
//!
//! ## Quick start
//!
//! The Gymnasium integration requires Python with Gymnasium installed.
//!
//! ```no_run
//! use r2l::PPOBuilder;
//!
//! let mut algorithm = PPOBuilder::gym("Pendulum-v1", 4)?.build()?;
//! algorithm.train().unwrap();
//! ```
//!
//! [`PPOBuilder`] and [`A2CBuilder`] are the main entry points for
//! configuring and training agents.
//!
//! For a more in-depth tutorial, see
//! [the Getting started chapter of the book](https://afgthecat.github.io/r2l/user_guide.html).
//! Complete workflows are available in the
//! [repository examples](https://github.com/afgTheCat/r2l/tree/main/crates/r2l-examples/examples).
//!
//! ## Current capabilities
//!
//! r2l currently provides:
//!
//! - on-policy training with PPO and A2C,
//! - Candle and Burn policy/value backends,
//! - a Gymnasium adapter,
//! - categorical, diagonal-Gaussian, multi-categorical, multi-Bernoulli, and
//!   composite policies,
//! - observation and reward normalization, evaluation and
//! - persistence of trained policies and inference
//!
//! ## Other crates
//!
//! r2l is the built on top of other crates.

#![warn(missing_docs)]
#![warn(unreachable_pub)]

use burn::backend::{Autodiff, NdArray};

// builders + hooks + higher level helpers
mod builders;
mod evaluator;
mod hooks;
mod utils;

/// Default autodifferentiation backend used by Burn-based builders.
pub type BurnBackend = Autodiff<NdArray>;

pub use builders::{
    A2CBuilder, A2CBurn, A2CCandle, AdamWParams, InferenceRunner, OnPolicyBuilder, PPOBuilder,
    PPOBurn, PPOCandle, TrainingArtifactsConfig,
};
pub use evaluator::EvaluationSettings;
pub use hooks::a2c::{A2CMinibatchStats, A2CRolloutStats};
pub use hooks::on_policy::{
    LearningRateSchedule, OnPolicyControlHandle, OnPolicyTrainingHooks, TrainingLimit,
};
pub use hooks::ppo::{PPOMinibatchStats, PPORolloutStats};
pub use hooks::sampler::{EpisodeBoundHook, StepBoundHook};
pub use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    models::ActivationFunction,
    on_policy::algorithm::OnPolicyAlgorithm,
    tensor::VecTensor,
};
pub use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};
