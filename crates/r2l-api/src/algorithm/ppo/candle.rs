use crate::{
    agents::ppo::{CandlePPO, PPOCandleLearningModuleBuilder},
    algorithm::OnPolicyAlgorightmBuilder,
    sampler::SamplerBuilder,
};
use r2l_core::{
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::LearningSchedule,
    sampler::buffer::StepTrajectoryBound,
};

pub type PPOCandleAlgorithmBuiler<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<CandlePPO, PPOCandleLearningModuleBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> PPOCandleAlgorithmBuiler<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: PPOCandleLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}
