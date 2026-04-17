use r2l_core::{
    env::{EnvBuilderTrait, Space},
    on_policy::algorithm::{Agent, OnPolicyAlgorithm},
};
use r2l_sampler::{FinalSampler, Location, StepTrajectoryBound, TrajectoryBound};

use crate::{
    builders::{agent::AgentBuilder, policy_builder::ActionSpaceType, sampler::SamplerBuilder},
    hooks::on_policy::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule},
};

type DefaultOnPolicyAlgorithm<A, EB, BD> = OnPolicyAlgorithm<
    A,
    FinalSampler<<EB as EnvBuilderTrait>::Env, BD>,
    DefaultOnPolicyAlgorightmsHooks<A, FinalSampler<<EB as EnvBuilderTrait>::Env, BD>>,
>;

pub struct OnPolicyAlgorightmBuilder<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor> = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>,
> {
    pub sampler_builder: SamplerBuilder<EB, BD>,
    pub learning_schedule: LearningSchedule,
    pub agent_builder: AB,
}

impl<A: Agent, AB: AgentBuilder<Agent = A>, EB: EnvBuilderTrait>
    OnPolicyAlgorightmBuilder<A, AB, EB>
{
    pub fn with_bound<BD2: TrajectoryBound<Tensor = EB::Tensor>>(
        self,
        trajectory_bound: BD2,
    ) -> OnPolicyAlgorightmBuilder<A, AB, EB, BD2> {
        let OnPolicyAlgorightmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            ..
        } = self;
        OnPolicyAlgorightmBuilder {
            sampler_builder: sampler_builder.with_bound(trajectory_bound),
            agent_builder,
            learning_schedule,
        }
    }
}

impl<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor>,
> OnPolicyAlgorightmBuilder<A, AB, EB, BD>
{
    pub fn with_location(mut self, location: Location) -> Self {
        self.sampler_builder = self.sampler_builder.with_location(location);
        self
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }

    pub fn build(self) -> anyhow::Result<DefaultOnPolicyAlgorithm<A, EB, BD>> {
        let env_description = self.sampler_builder.env_builder.env_description()?;
        let sampler = self.sampler_builder.build();
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space = match env_description.action_space {
            Space::Discrete(_) => ActionSpaceType::Discrete,
            Space::Continuous { .. } => ActionSpaceType::Continuous,
        };
        let agent = self
            .agent_builder
            .build(observation_size, action_size, action_space)?;
        let hooks = DefaultOnPolicyAlgorightmsHooks::new(self.learning_schedule);
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
        })
    }
}
