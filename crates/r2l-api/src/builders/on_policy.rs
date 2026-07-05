use std::marker::PhantomData;

use r2l_core::{
    env::{ActionSpaceType, EnvBuilder, Space},
    on_policy::algorithm::{
        Agent, DefaultAdapter, OnPolicyAdapters, OnPolicyAlgorithm, OnPolicyRuntime,
    },
    tensor::R2lTensor,
};
use r2l_sampler::{
    ClippedNormalizer, NormalizedSamplerHook, NormalizerMode, R2lNormalizedSampler, R2lSampler,
    SamplerExecutionMode,
};

use crate::{
    BestActorEvaluatorBuilder,
    builders::{
        agent::AgentBuilder,
        sampler::{SamplerBuilder, SamplerHookBuilder},
    },
    hooks::{
        on_policy::{DefaultOnPolicyAlgorithmHooks, LearningSchedule},
        sampler::EpisodeBoundHook,
    },
};

type DefaultOnPolicyAlgorithm<A, EB, SH> = OnPolicyAlgorithm<
    A,
    R2lSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
    DefaultOnPolicyAlgorithmHooks<
        A,
        R2lSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
        R2lSampler<<EB as EnvBuilder>::Env, EpisodeBoundHook<<EB as EnvBuilder>::Env>>,
    >,
>;

type NormalizedOnPolicyAlgorithm<A, EB, SH> = OnPolicyAlgorithm<
    A,
    R2lNormalizedSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
    DefaultOnPolicyAlgorithmHooks<
        A,
        R2lNormalizedSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
        R2lNormalizedSampler<<EB as EnvBuilder>::Env, EpisodeBoundHook<<EB as EnvBuilder>::Env>>,
    >,
>;

pub struct DirectSamplerSelection;

pub struct NormalizedSamplerSelection {
    obs_clip: f32,
}

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
    ST = DirectSamplerSelection,
> {
    pub(crate) sampler_builder: SamplerBuilder<EB, SH>,
    pub(crate) learning_schedule: LearningSchedule,
    pub(crate) evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    pub(crate) agent_builder: AB,
    pub(crate) sampler_type: ST,
    pub(crate) _phantom: PhantomData<A>,
}

impl<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env>,
    ST,
> OnPolicyAlgorithmBuilder<A, AB, EB, SH, ST>
{
    /// Creates an on-policy algorithm builder from an explicit sampler builder
    /// and agent builder.
    fn from_parts(
        sampler_builder: SamplerBuilder<EB, SH>,
        agent_builder: AB,
        sampler_type: ST,
    ) -> Self {
        Self {
            sampler_builder,
            agent_builder,
            evaluator_builder: None,
            learning_schedule: LearningSchedule::rollout_bound(300),
            sampler_type,
            _phantom: PhantomData,
        }
    }

    /// Replaces the sampler hook builder used to control rollout collection.
    pub fn with_hook<SH2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        hook_builder: SH2,
    ) -> OnPolicyAlgorithmBuilder<A, AB, EB, SH2, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
            sampler_type,
            _phantom,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(hook_builder),
            agent_builder,
            evaluator_builder,
            learning_schedule,
            sampler_type,
            _phantom: PhantomData,
        }
    }

    /// Replaces the rollout bound configuration by installing a new sampler
    /// hook builder.
    pub fn with_rollout_bound<SH2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        rollout_bound: SH2,
    ) -> OnPolicyAlgorithmBuilder<A, AB, EB, SH2, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
            sampler_type,
            _phantom,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(rollout_bound),
            agent_builder,
            evaluator_builder,
            learning_schedule,
            sampler_type,
            _phantom: PhantomData,
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

    /// Sets the frequency with which the evaluator runs
    pub fn with_evaluator_frequency(mut self, evauator_frequency: usize) -> Self {
        assert!(evauator_frequency > 0);
        let evaluator_builder = if let Some(evaluator_builder) = self.evaluator_builder.take() {
            evaluator_builder.with_evaluator_frequency(evauator_frequency)
        } else {
            let env_builder = self.sampler_builder.env_builder.clone();
            BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
                .with_evaluator_frequency(evauator_frequency)
        };
        self.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets how training environments are executed.
    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.sampler_builder = self.sampler_builder.with_execution_mode(location);
        self
    }
}

impl<A: Agent, AB: AgentBuilder<Agent = A>, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder<A, AB, EB, SH, DirectSamplerSelection>
{
    /// Creates an on-policy algorithm builder from an explicit sampler builder
    /// and agent builder.
    pub fn from_sampler_and_agent_builder(
        sampler_builder: SamplerBuilder<EB, SH>,
        agent_builder: AB,
    ) -> Self {
        Self::from_parts(sampler_builder, agent_builder, DirectSamplerSelection)
    }
}

impl<
    A: Agent,
    AB: AgentBuilder<Agent = A>,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env>,
    ST,
> OnPolicyAlgorithmBuilder<A, AB, EB, SH, ST>
{
    /// Switches training and evaluation rollout collection to normalized sampling.
    pub fn with_obs_normalizer(
        self,
        obs_clip: f32,
    ) -> OnPolicyAlgorithmBuilder<A, AB, EB, SH, NormalizedSamplerSelection> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
            ..
        } = self;
        OnPolicyAlgorithmBuilder::from_parts(
            sampler_builder,
            agent_builder,
            NormalizedSamplerSelection { obs_clip },
        )
        .with_learning_schedule(learning_schedule)
        .with_evaluator(evaluator_builder)
    }
}

impl<A: Agent, AB: AgentBuilder<Agent = A>, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder<A, AB, EB, SH, DirectSamplerSelection>
{
    /// Builds the configured on-policy algorithm runtime.
    pub fn build(self) -> anyhow::Result<DefaultOnPolicyAlgorithm<A, EB, SH>>
    where
        DefaultAdapter: OnPolicyAdapters<A::Actor, R2lSampler<<EB as EnvBuilder>::Env, SH::Target>>,
        <<EB as EnvBuilder>::Env as r2l_core::env::Env>::Tensor: R2lTensor,
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

impl<A: Agent, AB: AgentBuilder<Agent = A>, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder<A, AB, EB, SH, NormalizedSamplerSelection>
{
    /// Builds the configured on-policy algorithm runtime using normalized sampling.
    pub fn build(self) -> anyhow::Result<NormalizedOnPolicyAlgorithm<A, EB, SH>>
    where
        SH::Target: NormalizedSamplerHook<E = <EB as EnvBuilder>::Env>,
        DefaultAdapter:
            OnPolicyAdapters<A::Actor, R2lNormalizedSampler<<EB as EnvBuilder>::Env, SH::Target>>,
        <<EB as EnvBuilder>::Env as r2l_core::env::Env>::Tensor: R2lTensor,
    {
        let env_description = self.sampler_builder.env_builder.env_description()?;
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space = match env_description.action_space {
            Space::Discrete(_) => ActionSpaceType::Discrete,
            Space::Continuous { .. } => ActionSpaceType::Continuous,
        };
        let obs_size = env_description.observation_space.size();
        let obs_normalizer = ClippedNormalizer::new(
            NormalizerMode::Update,
            self.sampler_type.obs_clip,
            vec![obs_size],
        );
        let eval_obs_normalizer = obs_normalizer.with_mode(NormalizerMode::ReadOnly);
        let SamplerBuilder {
            env_builder,
            hook_builder,
            execution_mode,
        } = self.sampler_builder;
        let sampler = R2lNormalizedSampler::build_with_obs_normalizer(
            env_builder,
            hook_builder.build(),
            execution_mode,
            Some(obs_normalizer),
            false,
        );
        let agent = self
            .agent_builder
            .build(observation_size, action_size, action_space)?;
        let evaluator = self.evaluator_builder.map(|evaluator_builder| {
            let eval_sampler = R2lNormalizedSampler::build_with_obs_normalizer(
                evaluator_builder.env_builder().clone(),
                EpisodeBoundHook::new(evaluator_builder.n_episodes()),
                evaluator_builder.execution_mode(),
                Some(eval_obs_normalizer),
                false,
            );
            evaluator_builder.build_with_sampler(eval_sampler)
        });
        let hooks =
            DefaultOnPolicyAlgorithmHooks::with_evaluator(self.learning_schedule, evaluator);
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
