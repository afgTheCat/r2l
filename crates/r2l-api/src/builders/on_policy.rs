use r2l_core::{
    env::{ActionSpaceType, Env, EnvBuilder, Space},
    on_policy::algorithm::{
        Agent, DefaultAdapter, OnPolicyAdapters, OnPolicyAlgorithm, OnPolicyRuntime,
    },
    running_mean::RunningMeanStd2,
    tensor::RunningMeanTensor,
};
use r2l_sampler::{R2lSampler, SamplerExecutionMode};

use crate::{
    BestActorEvaluatorBuilder,
    builders::{
        agent::AgentBuilder,
        sampler::{SamplerBuilder, SamplerHookBuilder, StepHookBound},
    },
    hooks::on_policy::{DefaultOnPolicyAlgorithmHooks, LearningSchedule},
};

type DefaultOnPolicyAlgorithm<A, EB, SH> = OnPolicyAlgorithm<
    A,
    R2lSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
    DefaultOnPolicyAlgorithmHooks<
        A,
        R2lSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
    >,
>;

/// Generic builder for on-policy algorithms on the new training stack.
///
/// This builder combines:
/// - environment construction
/// - rollout collection via `SamplerBuilder`
/// - agent construction
/// - learning schedule configuration
/// - optional evaluation of the best actor during training
///
/// Algorithm-specific builders such as `PPOAlgorithmBuilder` and
/// `A2CAlgorithmBuilder` build on top of this type.
pub struct OnPolicyAlgorithmBuilder<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env>,
> {
    pub(crate) sampler_builder: SamplerBuilder<EB, SH>,
    pub(crate) learning_schedule: LearningSchedule,
    pub(crate) evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    pub(crate) agent_builder: AB,
}

impl<A: Agent, AB: AgentBuilder<Agent = A>, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder<A, AB, EB, SH>
{
    /// Creates an on-policy algorithm builder from an explicit sampler builder
    /// and agent builder.
    pub fn from_sampler_and_agent_builder(
        sampler_builder: SamplerBuilder<EB, SH>,
        agent_builder: AB,
    ) -> Self {
        Self {
            sampler_builder,
            agent_builder,
            evaluator_builder: None,
            learning_schedule: LearningSchedule::rollout_bound(300),
        }
    }

    /// Replaces the sampler hook builder used to control rollout collection.
    pub fn with_hook<SH2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        hook_builder: SH2,
    ) -> OnPolicyAlgorithmBuilder<A, AB, EB, SH2> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(hook_builder),
            agent_builder,
            evaluator_builder,
            learning_schedule,
        }
    }

    /// Replaces the rollout bound configuration by installing a new sampler
    /// hook builder.
    pub fn with_rollout_bound<SH2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        rollout_bound: SH2,
    ) -> OnPolicyAlgorithmBuilder<A, AB, EB, SH2> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(rollout_bound),
            agent_builder,
            evaluator_builder,
            learning_schedule,
        }
    }

    /// Installs or clears the evaluator used during training.
    pub fn with_evaluator(
        mut self,
        evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    ) -> Self {
        self.evaluator_builder = evaluator_builder;
        self
    }

    /// Replaces the learning schedule that controls training termination.
    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
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
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
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
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
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
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
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
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
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
    pub fn build(self) -> anyhow::Result<DefaultOnPolicyAlgorithm<A, EB, SH>>
    where
        DefaultAdapter: OnPolicyAdapters<A::Actor, R2lSampler<<EB as EnvBuilder>::Env, SH::Target>>,
        <<EB as EnvBuilder>::Env as r2l_core::env::Env>::Tensor: RunningMeanTensor,
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
            DefaultOnPolicyAlgorithmHooks::new(self.learning_schedule, self.evaluator_builder);
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

impl<A: Agent, AB: AgentBuilder<Agent = A>, EB: EnvBuilder<Env: Env<Tensor: RunningMeanTensor>>>
    OnPolicyAlgorithmBuilder<A, AB, EB, StepHookBound<EB::Env>>
{
    /// Enables observation normalization for step-bound rollout collection.
    pub fn with_observation_normalizer(mut self) -> Self {
        let observation_size = self
            .sampler_builder
            .env_builder
            .env_description()
            .expect("environment description should be available")
            .observation_size();
        let observation_normalizer = RunningMeanStd2::new(vec![observation_size]);
        self.sampler_builder.hook_builder = self
            .sampler_builder
            .hook_builder
            .with_observation_normalizer(observation_normalizer);
        self
    }
}
