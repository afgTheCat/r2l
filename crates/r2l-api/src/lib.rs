use burn::backend::{Autodiff, NdArray};

// builders + hooks + higher level helpers
mod agents;
mod builders;
mod hooks;
mod utils;

pub type BurnBackend = Autodiff<NdArray>;

pub use agents::a2c::{A2CBurnAgent, A2CCandleAgent};
pub use agents::ppo::{PPOBurnAgent, PPOCandleAgent};
pub use builders::a2c::agent::{A2CAgentBuilder, A2CBurnAgentBuilder, A2CCandleAgentBuilder};
pub use builders::a2c::algorithm::{A2CAlgorithmBuilder, A2CBurnAlgorithmBuilder, A2CCandleAlgorithmBuilder};
pub use builders::agent::{AgentBuilder, AgentBuilderStruct};
pub use builders::learning_module::LearningModuleType;
pub use builders::ppo::agent::{BurnPPOAgentBuilder, CandlePPOAgentBuilder, PPOAgentBuilder};
pub use builders::ppo::algorithm::{PPOAlgorithmBuilder, PPOBurnAlgorithmBuilder, PPOCandleAlgorithmBuilder};
pub use builders::sampler::SamplerBuilder;
pub use hooks::a2c::{A2CBatchStats, A2CStats, DefaultA2CHook, DefaultA2CHookReporter};
pub use hooks::on_policy::{DefaultOnPolicyAlgorithmsHooks, LearningSchedule};
pub use hooks::ppo::{BatchStats as PPOBatchStats, DefaultPPOHook, DefaultPPOHookReporter, PPOStats, TargetKl};
pub use hooks::sampler::{
    EnvNormalizer, EvaluatorNormalizer, EvaluatorNormalizerOptions, EvaluatorOptions,
    NormalizerOptions,
};
pub use r2l_core::on_policy::algorithm::OnPolicyAlgorithm;
pub use r2l_sampler::{EpisodeTrajectoryBound, Location, R2lSampler, StepTrajectoryBound};
