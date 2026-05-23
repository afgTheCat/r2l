use burn::backend::{Autodiff, NdArray};

// builders + hooks + higher level helpers
mod agents;
mod builders;
mod hooks;
mod utils;

type BurnBackend = Autodiff<NdArray>;

pub use agents::a2c::{A2CBurnAgent, A2CCandleAgent};
pub use agents::ppo::{PPOBurnAgent, PPOCandleAgent};
pub use agents::ppo2::{PPO2BurnAgent, PPO2CandleAgent};
pub use builders::a2c::agent::{A2CAgentBuilder, A2CBurnAgentBuilder, A2CCandleAgentBuilder};
pub use builders::a2c::algorithm::{
    A2CAlgorithmBuilder, A2CBurnAlgorithmBuilder, A2CCandleAlgorithmBuilder,
};
pub use builders::agent::OnPolicyAgentBuilder;
pub use builders::agent2::OnPolicyAgentBuilder2;
pub use builders::learning_module::OnPolicyLearningModuleType;
pub use builders::on_policy::OnPolicyAlgorithmBuilder;
pub use builders::on_policy2::OnPolicyAlgorithmBuilder2;
pub use builders::ppo::agent::{PPOAgentBuilder, PPOBurnAgentBuilder, PPOCandleAgentBuilder};
pub use builders::ppo::agent2::{PPO2AgentBuilder, PPO2BurnAgentBuilder, PPO2CandleAgentBuilder};
pub use builders::ppo::algorithm::{
    PPOAlgorithmBuilder, PPOBurnAlgorithmBuilder, PPOCandleAlgorithmBuilder,
};
pub use builders::ppo::algorithm2::{
    PPO2AlgorithmBuilder, PPO2BurnAlgorithmBuilder, PPO2CandleAlgorithmBuilder,
};
pub use builders::sampler::SamplerBuilder;
pub use hooks::a2c::{A2CBatchStats, A2CStats, DefaultA2CHook};
pub use hooks::on_policy::{DefaultOnPolicyAlgorithmHooks, LearningSchedule};
pub use hooks::on_policy2::{DefaultOnPolicyAlgorithmHooks2, LearningSchedule2};
pub use hooks::ppo::{DefaultPPOHook, PPOBatchStats, PPOStats};
pub use hooks::ppo2::{DefaultPPO2Hook, PPO2BatchStats, PPO2Stats};
pub use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    on_policy::algorithm::OnPolicyAlgorithm,
    tensor::TensorData,
};
pub use r2l_sampler::{
    EpisodeTrajectoryBound, R2lSampler, SamplerExecutionMode, StepTrajectoryBound,
};
pub use utils::evaluator::{BestActorEvaluator, BestActorEvaluatorBuilder, Evaluator};
