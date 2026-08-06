//! High-level builders and default hooks for training `r2l` agents.
//!
//! Most users should start with [`PPOAlgorithmBuilder`] or
//! [`A2CAlgorithmBuilder`]. These builders combine an environment, sampler,
//! backend-specific agent, training schedule, and optional evaluator.
//!
//! ```no_run
//! use r2l::{LearningSchedule, PPOAlgorithmBuilder};
//!
//! let mut algorithm = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
//!     .with_rollout_steps(1024)
//!     .with_learning_schedule(LearningSchedule::total_step_bound(100_000))
//!     .build()
//!     .unwrap();
//! algorithm.train().unwrap();
//! ```

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
    PolicyBuilder,
};
pub use evaluators::best_actor_evaluator::{EvaluationSettings, TrainingArtifactsConfig};
pub use evaluators::simple_evaluator::Evaluator;
pub use hooks::a2c::{A2CBatchStats, A2CStats};
pub use hooks::on_policy::{
    DefaultOnPolicyAlgorithmHooks, LearningRateSchedule, LearningSchedule, OnPolicyCommand,
    OnPolicyCommandReceiver, OnPolicyCommandResult, OnPolicyCommandSender,
    on_policy_command_channel,
};
pub use hooks::ppo::{PPOBatchStats, PPOStats};
pub use hooks::sampler::{EpisodeBoundHook, StepBoundHook};
pub use r2l_core::{
    env::{
        Env, EnvBuilder, EnvDescription, Snapshot, Space,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    models::ActivationFunction,
    on_policy::algorithm::OnPolicyAlgorithm,
    tensor::TensorData,
};
pub use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};
