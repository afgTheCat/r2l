use r2l_core::{
    env::{ActionSpaceType, EnvBuilder, Space},
    on_policy::algorithm2::{
        Agent2, DefaultAdapter, OnPolicyAdapters2, OnPolicyAlgorithm, OnPolicyRuntime,
    },
};
use r2l_sampler::{
    SamplerExecutionMode,
    sampler2::R2lSampler2,
};

use crate::{
    BestActorEvaluatorBuilder2,
    builders::{agent2::AgentBuilder2, sampler2::{Sampler2Builder, SamplerHook2Builder}},
    hooks::on_policy2::{DefaultOnPolicyAlgorithmHooks2, LearningSchedule2},
};

type DefaultOnPolicyAlgorithm2<A, EB, SH> = OnPolicyAlgorithm<
    A,
    R2lSampler2<<EB as EnvBuilder>::Env, <SH as SamplerHook2Builder>::Target>,
    DefaultOnPolicyAlgorithmHooks2<
        A,
        R2lSampler2<<EB as EnvBuilder>::Env, <SH as SamplerHook2Builder>::Target>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
    >,
>;

/// Generic builder for on-policy algorithms on the new training stack.
///
/// This builder combines:
/// - environment construction
/// - rollout collection via `Sampler2Builder`
/// - agent construction
/// - learning schedule configuration
/// - optional evaluation of the best actor during training
///
/// Algorithm-specific builders such as `PPO2AlgorithmBuilder` and
/// `A2C2AlgorithmBuilder` build on top of this type.
pub struct OnPolicyAlgorithmBuilder2<
    A: Agent2,
    AB: AgentBuilder2<Agent = A>,
    EB: EnvBuilder,
    SH: SamplerHook2Builder<Env = EB::Env>,
> {
    pub(crate) sampler_builder: Sampler2Builder<EB, SH>,
    pub(crate) learning_schedule: LearningSchedule2,
    pub(crate) evaluator_builder: Option<BestActorEvaluatorBuilder2<EB>>,
    pub(crate) agent_builder: AB,
}

impl<
    A: Agent2,
    AB: AgentBuilder2<Agent = A>,
    EB: EnvBuilder,
    SH: SamplerHook2Builder<Env = EB::Env>,
> OnPolicyAlgorithmBuilder2<A, AB, EB, SH>
{
    /// Creates an on-policy algorithm builder from an explicit sampler builder
    /// and agent builder.
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

    /// Replaces the sampler hook builder used to control rollout collection.
    pub fn with_hook<SH2: SamplerHook2Builder<Env = EB::Env>>(
        self,
        hook_builder: SH2,
    ) -> OnPolicyAlgorithmBuilder2<A, AB, EB, SH2> {
        let OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
        } = self;
        OnPolicyAlgorithmBuilder2 {
            sampler_builder: sampler_builder.with_hook(hook_builder),
            agent_builder,
            evaluator_builder,
            learning_schedule,
        }
    }

    /// Replaces the rollout bound configuration by installing a new sampler
    /// hook builder.
    pub fn with_rollout_bound<SH2: SamplerHook2Builder<Env = EB::Env>>(
        self,
        rollout_bound: SH2,
    ) -> OnPolicyAlgorithmBuilder2<A, AB, EB, SH2> {
        let OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
        } = self;
        OnPolicyAlgorithmBuilder2 {
            sampler_builder: sampler_builder.with_hook(rollout_bound),
            agent_builder,
            evaluator_builder,
            learning_schedule,
        }
    }

    /// Installs or clears the evaluator used during training.
    pub fn with_evaluator(
        mut self,
        evaluator_builder: Option<BestActorEvaluatorBuilder2<EB>>,
    ) -> Self {
        self.evaluator_builder = evaluator_builder;
        self
    }

    /// Replaces the learning schedule that controls training termination.
    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule2) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }

    /// Sets the number of evaluation episodes used by the best-actor
    /// evaluator.
    pub fn with_evaluator_n_episodes(mut self, n_episodes: usize) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_n_episodes(n_episodes)
        } else {
            let env_builder = self.sampler_builder.env_builder.clone();
            BestActorEvaluatorBuilder2::from_env_builder_type(env_builder)
                .with_n_episodes(n_episodes)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Replaces the environment builder used by the evaluator.
    pub fn with_evaluator_env_builder(
        mut self,
        env_builder: r2l_core::env::EnvBuilderType<EB>,
    ) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_env_builder(env_builder)
        } else {
            BestActorEvaluatorBuilder2::from_env_builder_type(env_builder)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets how evaluation environments are executed.
    pub fn with_evaluator_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_execution_mode(execution_mode)
        } else {
            let env_builder = self.sampler_builder.env_builder.clone();
            BestActorEvaluatorBuilder2::from_env_builder_type(env_builder)
                .with_execution_mode(execution_mode)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets the filesystem path used to persist the best-performing actor.
    pub fn with_evaluator_best_actor_path<P: Into<std::path::PathBuf>>(
        mut self,
        eval_path: P,
    ) -> Self {
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_best_actor_path(eval_path)
        } else {
            let env_builder = self.sampler_builder.env_builder.clone();
            BestActorEvaluatorBuilder2::from_env_builder_type(env_builder)
                .with_best_actor_path(eval_path)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets how training environments are executed.
    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.sampler_builder = self.sampler_builder.with_execution_mode(location);
        self
    }

    /// Builds the configured on-policy algorithm runtime.
    pub fn build(self) -> anyhow::Result<DefaultOnPolicyAlgorithm2<A, EB, SH>>
    where
        DefaultAdapter:
            OnPolicyAdapters2<A::Actor, R2lSampler2<<EB as EnvBuilder>::Env, SH::Target>>,
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
