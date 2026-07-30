//! High-level builders and default hooks for training `r2l` agents.
//!
//! Most users should start with [`PPOAlgorithmBuilder`] or
//! [`A2CAlgorithmBuilder`]. These builders combine an environment, sampler,
//! backend-specific agent, training schedule, and optional evaluator.

use burn::backend::{Autodiff, NdArray};

// builders + hooks + higher level helpers
mod agents;
mod builders;
mod evaluators;
mod hooks;
mod utils;

/// Default autodifferentiation backend used by Burn-based builders.
pub type BurnBackend = Autodiff<NdArray>;

pub use agents::a2c::{A2CBurnAgent, A2CCandleAgent};
pub use agents::ppo::{PPOBurnAgent, PPOCandleAgent};
pub use builders::a2c::agent::{A2CBurnAgentBuilder, A2CCandleAgentBuilder};
pub use builders::a2c::algorithm::{
    A2CAlgorithmBuilder, A2CBurnAlgorithmBuilder, A2CCandleAlgorithmBuilder,
};
pub use builders::agent::{BurnBackendConfig, CandleBackend, OnPolicyAgentBuilder};
// pub use builders::inference::{
//     BurnInferenceRunner, CandleInferenceRunner, InferenceRunner, InferenceRunnerBuilder,
// };
pub use builders::learning_module::OnPolicyOptimizerLayout;
pub use builders::on_policy::OnPolicyAlgorithmBuilder;
pub use builders::policy::PolicyBuilder;
pub use builders::ppo::agent::{PPOAgentBuilder, PPOBurnAgentBuilder, PPOCandleAgentBuilder};
pub use builders::ppo::algorithm::{
    PPOAlgorithmBuilder, PPOBurnAlgorithmBuilder, PPOCandleAlgorithmBuilder,
};
pub use builders::sampler::{
    BuiltSampler, ConfiguredSamplerBuilder, DefaultSamplerBuilder, DirectSamplerSelection,
    SamplerBuilder, StagedSamplerSelection,
};
pub use builders::sampler::{EpisodeHookBound, StepHookBound};
pub use evaluators::best_actor_evaluator::{BestActorEvaluator, BestActorEvaluatorBuilder};
pub use evaluators::simple_evaluator::Evaluator;
pub use hooks::a2c::{A2CBatchStats, A2CStats, DefaultA2CHook};
pub use hooks::on_policy::{
    DefaultOnPolicyAlgorithmHooks, LearningRateSchedule, LearningSchedule, OnPolicyCommand,
    OnPolicyCommandReceiver, OnPolicyCommandResult, OnPolicyCommandSender,
    on_policy_command_channel,
};
pub use hooks::ppo::{DefaultPPOHook, PPOBatchStats, PPOStats};
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
