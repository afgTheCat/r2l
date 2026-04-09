use crate::sampler::SamplerBuilder;
use crate::{
    agents::ppo::{BurnBackend, BurnPPO, PPOBurnLearningModuleBuilder},
    algorithm::OnPolicyAlgorightmBuilder,
};
use r2l_core::{
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::LearningSchedule,
    sampler::buffer::StepTrajectoryBound,
};

pub type PPOBurnAlgorithmBuiler<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<BurnPPO<BurnBackend>, PPOBurnLearningModuleBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> PPOBurnAlgorithmBuiler<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: PPOBurnLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}
