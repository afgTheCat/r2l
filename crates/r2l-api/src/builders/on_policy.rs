use std::path::PathBuf;

use r2l_core::{
    env::{ActionSpaceType, EnvBuilder, Space, TensorOfEnvBuilder},
    on_policy::algorithm::{
        Agent, DefaultAdapter, OnPolicyAdapters, OnPolicyAlgorithm, OnPolicyRuntime,
    },
};
use r2l_sampler::{R2lSampler, SamplerExecutionMode, StepTrajectoryBound, TrajectoryBound};

use crate::{
    builders::{agent::AgentBuilder, sampler::SamplerBuilder},
    hooks::on_policy::{DefaultOnPolicyAlgorithmHooks, LearningSchedule},
};

type DefaultOnPolicyAlgorithm<A, EB, BD> = OnPolicyAlgorithm<
    A,
    R2lSampler<<EB as EnvBuilder>::Env, BD>,
    DefaultOnPolicyAlgorithmHooks<
        A,
        R2lSampler<<EB as EnvBuilder>::Env, BD>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
    >,
>;

pub struct OnPolicyAlgorithmBuilder<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilder,
    BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>> = StepTrajectoryBound<
        TensorOfEnvBuilder<EB>,
    >,
> {
    pub(crate) sampler_builder: SamplerBuilder<EB, BD>,
    pub(crate) learning_schedule: LearningSchedule,
    pub(crate) eval_env_builder: Option<EB>,
    pub(crate) eval_model_path: Option<PathBuf>,
    pub(crate) agent_builder: AB,
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
            eval_env_builder: None,
            eval_model_path: None,
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
            eval_model_path,
            ..
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_bound(trajectory_bound),
            agent_builder,
            eval_env_builder: None,
            eval_model_path,
            learning_schedule,
        }
    }

    pub fn with_eval_env(mut self, eval_env_builder: EB) -> Self {
        self.eval_env_builder = Some(eval_env_builder);
        self
    }

    pub fn with_eval_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.eval_model_path = Some(path.into());
        self
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }

    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.sampler_builder = self.sampler_builder.with_execution_mode(location);
        self
    }

    pub fn build(self) -> anyhow::Result<DefaultOnPolicyAlgorithm<A, EB, BD>>
    where
        DefaultAdapter: OnPolicyAdapters<A, R2lSampler<<EB as EnvBuilder>::Env, BD>>,
    {
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
        let hooks = DefaultOnPolicyAlgorithmHooks::new(
            self.learning_schedule,
            self.eval_env_builder,
            self.eval_model_path,
        );
        Ok(OnPolicyAlgorithm {
            runtime: OnPolicyRuntime {
                sampler,
                agent,
                adapter: DefaultAdapter,
            },
            hooks,
        })
    }
}
