use r2l_core::{
    env::{ActionSpaceType, EnvBuilder, Space},
    on_policy::algorithm2::{
        Agent2, DefaultAdapter, OnPolicyAdapters2, OnPolicyAlgorithm, OnPolicyRuntime,
    },
};
use r2l_sampler::{SamplerExecutionMode, sampler2::R2lSampler2};

use crate::{
    BestActorEvaluatorBuilder,
    builders::{agent2::AgentBuilder2, sampler2::Sampler2Builder},
    hooks::on_policy2::{DefaultOnPolicyAlgorithmHooks2, LearningSchedule2},
};

type DefaultOnPolicyAlgorithm2<A, EB, SH> = OnPolicyAlgorithm<
    A,
    R2lSampler2<<EB as EnvBuilder>::Env, SH>,
    DefaultOnPolicyAlgorithmHooks2<
        A,
        R2lSampler2<<EB as EnvBuilder>::Env, SH>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
    >,
>;

pub struct OnPolicyAlgorithmBuilder2<
    A: Agent2,
    AB: AgentBuilder2<Agent = A>,
    EB: EnvBuilder,
    SH: r2l_sampler::sampler2::SamplerHook2<E = EB::Env>,
> {
    pub(crate) sampler_builder: Sampler2Builder<EB, SH>,
    pub(crate) learning_schedule: LearningSchedule2,
    pub(crate) evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    pub(crate) agent_builder: AB,
}

impl<A: Agent2, AB: AgentBuilder2<Agent = A>, EB: EnvBuilder, SH: r2l_sampler::sampler2::SamplerHook2<E = EB::Env>>
    OnPolicyAlgorithmBuilder2<A, AB, EB, SH>
{
    pub fn from_sampler_and_agent_builder(
        sampler_builder: Sampler2Builder<EB, SH>,
        agent_builder: AB,
    ) -> Self {
        Self {
            sampler_builder,
            agent_builder,
            evaluator_builder: None,
            learning_schedule: LearningSchedule2::rollout_bound(300),
        }
    }

    pub fn with_hook<SH2: r2l_sampler::sampler2::SamplerHook2<E = EB::Env>>(
        self,
        hook: SH2,
    ) -> OnPolicyAlgorithmBuilder2<A, AB, EB, SH2> {
        let OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
        } = self;
        OnPolicyAlgorithmBuilder2 {
            sampler_builder: sampler_builder.with_hook(hook),
            agent_builder,
            evaluator_builder,
            learning_schedule,
        }
    }

    pub fn with_evaluator(
        mut self,
        evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    ) -> Self {
        self.evaluator_builder = evaluator_builder;
        self
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule2) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }

    pub fn with_evaluator_n_episodes(mut self, n_episodes: usize) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_n_episodes(n_episodes)
        } else {
            let env_builder = self.sampler_builder.env_builder.clone();
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder).with_n_episodes(n_episodes)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    pub fn with_evaluator_env_builder(
        mut self,
        env_builder: r2l_core::env::EnvBuilderType<EB>,
    ) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_env_builder(env_builder)
        } else {
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    pub fn with_evaluator_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_execution_mode(execution_mode)
        } else {
            let env_builder = self.sampler_builder.env_builder.clone();
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder).with_execution_mode(execution_mode)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    pub fn with_evaluator_best_actor_path<P: Into<std::path::PathBuf>>(
        mut self,
        eval_path: P,
    ) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_best_actor_path(eval_path)
        } else {
            let env_builder = self.sampler_builder.env_builder.clone();
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder).with_best_actor_path(eval_path)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.sampler_builder = self.sampler_builder.with_execution_mode(location);
        self
    }

    pub fn build(self) -> anyhow::Result<DefaultOnPolicyAlgorithm2<A, EB, SH>>
    where
        DefaultAdapter: OnPolicyAdapters2<A::Actor, R2lSampler2<<EB as EnvBuilder>::Env, SH>>,
        A::Tensor: From<<EB::Env as r2l_core::env::Env>::Tensor>,
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
        let hooks =
            DefaultOnPolicyAlgorithmHooks2::new(self.learning_schedule, self.evaluator_builder);
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
