use r2l_core::{
    env::{ActionSpaceType, EnvBuilder, Space, TensorOfEnvBuilder},
    on_policy::algorithm::{Agent, OnPolicyAlgorithm},
};
use r2l_sampler::{Location, R2lSampler, StepTrajectoryBound, TrajectoryBound};

use crate::{
    builders::{agent::AgentBuilder, sampler::SamplerBuilder},
    hooks::on_policy::{DefaultOnPolicyAlgorithmsHooks, LearningSchedule},
};

type DefaultOnPolicyAlgorithm<A, EB, BD> = OnPolicyAlgorithm<
    A,
    R2lSampler<<EB as EnvBuilder>::Env, BD>,
    DefaultOnPolicyAlgorithmsHooks<A, R2lSampler<<EB as EnvBuilder>::Env, BD>>,
>;

pub struct OnPolicyAlgorithmBuilder<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilder,
    BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>> = StepTrajectoryBound<
        TensorOfEnvBuilder<EB>,
    >,
> {
    pub sampler_builder: SamplerBuilder<EB, BD>,
    pub learning_schedule: LearningSchedule,
    pub agent_builder: AB,
}

impl<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilder,
    BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>>,
> OnPolicyAlgorithmBuilder<A, AB, EB, BD>
{
    pub fn from_sampler_and_agent_builder(
        sampler_builder: SamplerBuilder<EB, BD>,
        agent_builder: AB,
    ) -> Self {
        Self {
            sampler_builder,
            agent_builder,
            learning_schedule: LearningSchedule::rollout_bound(300),
        }
    }

    pub fn with_bound<BD2: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>>>(
        self,
        trajectory_bound: BD2,
    ) -> OnPolicyAlgorithmBuilder<A, AB, EB, BD2> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            ..
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_bound(trajectory_bound),
            agent_builder,
            learning_schedule,
        }
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.sampler_builder = self.sampler_builder.with_location(location);
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
        let hooks = DefaultOnPolicyAlgorithmsHooks::new(self.learning_schedule);
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
        })
    }
}
