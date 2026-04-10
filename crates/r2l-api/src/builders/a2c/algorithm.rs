use crate::{
    agents::{
        a2c::{BurnA2C, CandleA2C},
    },
    builders::{
        a2c::agent::{A2CAgentBuilder, A2CBurnLearningModuleBuilder, A2CCandleLearningModuleBuilder},
        agent::AgentBuilder,
        on_policy::OnPolicyAlgorightmBuilder,
        sampler::SamplerBuilder,
    },
    hooks::on_policy::LearningSchedule,
    BurnBackend,
};
use r2l_core::{
    agents::Agent,
    env_builder::EnvBuilderTrait,
    sampler::buffer::{StepTrajectoryBound, TrajectoryBound},
};

impl<A, M, EB, BD> OnPolicyAlgorightmBuilder<A, A2CAgentBuilder<M>, EB, BD>
where
    A: Agent,
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor>,
    A2CAgentBuilder<M>: AgentBuilder<Agent = A>,
{
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.agent_builder.a2c_params.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.agent_builder.a2c_params.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.agent_builder.a2c_params.sample_size = sample_size;
        self
    }
}

pub type A2CBurnAlgorithmBuilder<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<BurnA2C<BurnBackend>, A2CBurnLearningModuleBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> A2CBurnAlgorithmBuilder<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: A2CBurnLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}

pub type A2CCandleAlgorithmBuilder<
    EB,
    BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>,
> = OnPolicyAlgorightmBuilder<CandleA2C, A2CCandleLearningModuleBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> A2CCandleAlgorithmBuilder<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: A2CCandleLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}
