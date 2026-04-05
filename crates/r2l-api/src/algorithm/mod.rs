pub mod ppo_algorithm;

use r2l_core::{
    agents::Agent,
    env::Space,
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
    sampler::{
        FinalSampler,
        buffer::{StepTrajectoryBound, TrajectoryBound},
    },
};

use crate::{
    agents::AgentBuilder, builders::distribution::ActionSpaceType, sampler::SamplerBuilder,
};

pub struct AlgorightmBuilder<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor> = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>,
> {
    pub sampler_builder: SamplerBuilder<EB, BD>,
    pub learning_schedule: LearningSchedule,
    pub agent_builder: AB,
}

impl<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor>,
> AlgorightmBuilder<A, AB, EB, BD>
{
    pub fn build(
        self,
    ) -> anyhow::Result<
        OnPolicyAlgorithm<
            A,
            FinalSampler<EB::Env, BD>,
            DefaultOnPolicyAlgorightmsHooks<A, FinalSampler<EB::Env, BD>>,
        >,
    > {
        let env_description = self.sampler_builder.env_builder.env_description()?;
        let sampler = self.sampler_builder.build();
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space = match env_description.action_space {
            Space::Discrete(_) => ActionSpaceType::Discrete,
            Space::Continous { .. } => ActionSpaceType::Continous,
        };
        let agent = self
            .agent_builder
            .build(observation_size, action_size, action_space)?;
        let hooks = DefaultOnPolicyAlgorightmsHooks::new(LearningSchedule::RolloutBound {
            total_rollouts: 300,
            current_rollout: 0,
        });
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
        })
    }
}
