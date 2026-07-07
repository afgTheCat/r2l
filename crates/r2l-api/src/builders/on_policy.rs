use r2l_core::{
    env::{ActionSpaceType, EnvBuilder, Space},
    on_policy::algorithm::{
        Agent, DefaultAdapter, OnPolicyAdapters, OnPolicyAlgorithm, OnPolicyRuntime,
    },
    rng::{sample_u64, set_seed},
    tensor::R2lTensor,
};
use r2l_sampler::{
    NormalizedSamplerHook, NormalizerMode, R2lNormalizedSampler, R2lSampler, SamplerExecutionMode,
};

use crate::{
    BestActorEvaluatorBuilder,
    builders::{
        agent::AgentBuilder,
        sampler::{
            DirectSamplerSelection, NormalizedSamplerSelection, SamplerBuilder, SamplerHookBuilder,
        },
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

type DefaultOnPolicyAlgorithmFor<AB, EB, SH> =
    DefaultOnPolicyAlgorithm<<AB as AgentBuilder>::Agent, EB, SH>;

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

type NormalizedOnPolicyAlgorithmFor<AB, EB, SH> =
    NormalizedOnPolicyAlgorithm<<AB as AgentBuilder>::Agent, EB, SH>;

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
    AB: AgentBuilder,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env>,
    ST = DirectSamplerSelection,
> {
    pub(crate) sampler_builder: SamplerBuilder<EB, SH, ST>,
    pub(crate) learning_schedule: LearningSchedule,
    pub(crate) evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    pub(crate) agent_builder: AB,
    pub(crate) seed: Option<u64>,
}

impl<AB: AgentBuilder, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>, ST>
    OnPolicyAlgorithmBuilder<AB, EB, SH, ST>
{
    /// Creates an on-policy algorithm builder from an explicit sampler builder
    /// and agent builder.
    fn from_parts(sampler_builder: SamplerBuilder<EB, SH, ST>, agent_builder: AB) -> Self {
        Self {
            sampler_builder,
            agent_builder,
            evaluator_builder: None,
            learning_schedule: LearningSchedule::rollout_bound(300),
            seed: None,
        }
    }

    /// Replaces the sampler hook builder used to control rollout collection.
    pub fn with_hook<SH2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        hook_builder: SH2,
    ) -> OnPolicyAlgorithmBuilder<AB, EB, SH2, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(hook_builder),
            agent_builder,
            evaluator_builder,
            learning_schedule,
            seed,
        }
    }

    /// Replaces the rollout bound configuration by installing a new sampler
    /// hook builder.
    pub fn with_rollout_bound<SH2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        rollout_bound: SH2,
    ) -> OnPolicyAlgorithmBuilder<AB, EB, SH2, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            evaluator_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(rollout_bound),
            agent_builder,
            evaluator_builder,
            learning_schedule,
            seed,
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

    /// Sets the seed used by r2l, Gym reset seeds, and backend-specific RNGs.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

    /// Switches training and evaluation rollout collection to normalized observations.
    pub fn with_observation_normalizer(
        self,
        obs_clip: f32,
    ) -> OnPolicyAlgorithmBuilder<AB, EB, SH, NormalizedSamplerSelection> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_obs_normalizer(obs_clip),
            learning_schedule,
            evaluator_builder,
            agent_builder,
            seed,
        }
    }
}

impl<AB: AgentBuilder, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder<AB, EB, SH, DirectSamplerSelection>
{
    /// Creates an on-policy algorithm builder from an explicit sampler builder
    /// and agent builder.
    pub fn from_sampler_and_agent_builder(
        sampler_builder: SamplerBuilder<EB, SH, DirectSamplerSelection>,
        agent_builder: AB,
    ) -> Self {
        Self::from_parts(sampler_builder, agent_builder)
    }
}

impl<AB: AgentBuilder, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder<AB, EB, SH, DirectSamplerSelection>
{
    /// Builds the configured on-policy algorithm runtime.
    pub fn build(self) -> anyhow::Result<DefaultOnPolicyAlgorithmFor<AB, EB, SH>>
    where
        DefaultAdapter: OnPolicyAdapters<
                <<AB as AgentBuilder>::Agent as Agent>::Actor,
                R2lSampler<<EB as EnvBuilder>::Env, SH::Target>,
            >,
    {
        if let Some(seed) = self.seed {
            set_seed(seed);
        }
        let env_description = self.sampler_builder.env_builder.env_description()?;
        let sampler = self.sampler_builder.build();
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space = match env_description.action_space {
            Space::Discrete(_) => ActionSpaceType::Discrete,
            Space::Continuous { .. } => ActionSpaceType::Continuous,
        };
        let agent =
            self.agent_builder
                .build(observation_size, action_size, action_space, self.seed)?;
        let evaluator = self.evaluator_builder.map(|eb| eb.build());
        let hooks = DefaultOnPolicyAlgorithmHooks::new(self.learning_schedule, evaluator);
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

impl<
    AB: AgentBuilder,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env, Target: NormalizedSamplerHook<E = <EB as EnvBuilder>::Env>>,
> OnPolicyAlgorithmBuilder<AB, EB, SH, NormalizedSamplerSelection>
{
    /// Builds the configured on-policy algorithm runtime using normalized sampling.
    pub fn build(self) -> anyhow::Result<NormalizedOnPolicyAlgorithmFor<AB, EB, SH>>
    where
        DefaultAdapter: OnPolicyAdapters<
                <<AB as AgentBuilder>::Agent as Agent>::Actor,
                R2lNormalizedSampler<<EB as EnvBuilder>::Env, SH::Target>,
            >,
    {
        if let Some(seed) = self.seed {
            set_seed(seed);
        }
        let env_description = self.sampler_builder.env_builder.env_description()?;
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space = match env_description.action_space {
            Space::Discrete(_) => ActionSpaceType::Discrete,
            Space::Continuous { .. } => ActionSpaceType::Continuous,
        };
        let sampler = self.sampler_builder.build();
        let eval_obs_normalizer = sampler.obs_normalizer(NormalizerMode::ReadOnly);
        let agent =
            self.agent_builder
                .build(observation_size, action_size, action_space, self.seed)?;
        let evaluator = self.evaluator_builder.map(|evaluator_builder| {
            let eval_sampler = R2lNormalizedSampler::build_with_obs_normalizer(
                evaluator_builder.env_builder().clone(),
                EpisodeBoundHook::new(evaluator_builder.n_episodes()),
                evaluator_builder.execution_mode(),
                eval_obs_normalizer,
                false,
            );
            evaluator_builder.build_with_sampler(eval_sampler)
        });
        let hooks = DefaultOnPolicyAlgorithmHooks::new(self.learning_schedule, evaluator);
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
