//! r2l is a reinforcement learning library for Rust. It provides high-level
//! [Proximal Policy Optimization (PPO)][ppo] and [Advantage Actor-Critic (A2C)][a2c]
//! training builders while exposing lower-level components for customization.
//!
//! [ppo]: https://arxiv.org/abs/1707.06347
//! [a2c]: https://openai.com/index/openai-baselines-acktr-a2c/
//!
//! ## Quick start
//!
//! The Gymnasium integration is available through the opt-in `gym` feature. It
//! requires Python 3.11 or newer with the `gymnasium` package installed.
//!
//! ```no_run
//! use r2l::PPOBuilder;
//!
//! # #[cfg(feature = "gym")]
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut algorithm = PPOBuilder::gym("Pendulum-v1", 4)?.build()?;
//!     algorithm.train()?;
//!     Ok(())
//! }
//! # #[cfg(not(feature = "gym"))]
//! # fn main() {}
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
//! - observation and reward normalization,
//! - evaluation and persistence of trained policies, and
//! - inference from saved policies.
//!
//! ## Other crates
//!
//! `r2l` is built on top of these lower-level crates:
//!
//! - [r2l-core](https://docs.rs/r2l-core) — core traits and data types.
//! - [r2l-sampler](https://docs.rs/r2l-sampler) — rollout samplers.
//! - [r2l-gym](https://docs.rs/r2l-gym) — Gymnasium-backed environments.
//! - [r2l-burn](https://docs.rs/r2l-burn) — Burn-backed policy and learner implementations.
//! - [r2l-candle](https://docs.rs/r2l-candle) — Candle-backed policy and learner implementations.
//! - [r2l-agents](https://docs.rs/r2l-agents) — core RL algorithm implementations.

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
    A2CBuilder, A2CBurn, A2CCandle, AdamWParams, InferenceEnv, InferencePolicy, InferenceRunner,
    OnPolicyBuilder, PPOBuilder, PPOBurn, PPOCandle, TrainingArtifactsConfig,
};
pub use evaluator::EvaluationSettings;
pub use hooks::a2c::{A2CMinibatchStats, A2CRolloutStats};
pub use hooks::on_policy::{
    LearningRateSchedule, OnPolicyControlHandle, OnPolicyTrainingHooks, TrainingLimit,
};
pub use hooks::ppo::{ClipRangeSchedule, PPOMinibatchStats, PPORolloutStats};
pub use hooks::sampler::{EpisodeBoundHook, StepBoundHook};
pub use r2l_core::error::{self, Error};
pub use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    models::ActivationFunction,
    on_policy::algorithm::OnPolicyAlgorithm,
    tensor::VecTensor,
};
#[cfg(feature = "gym")]
pub use r2l_gym::{GymEnv, GymEnvBuilder};
pub use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};
