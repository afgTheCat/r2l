use r2l_core::{
    env::{
        Env, EnvBuilder,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    on_policy::algorithm::{
        Agent, DefaultAdapter, OnPolicyAdapters, OnPolicyAlgorithm, OnPolicyRuntime, Sampler,
    },
    rng::set_seed,
    tensor::R2lTensor,
};
use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler, StagedSamplerHook};
use serde::{Deserialize, Serialize};

use crate::{
    BestActorEvaluatorBuilder, DefaultOnPolicyAlgorithmHooks, OnPolicyCommandReceiver,
    builders::{
        agent::AgentBuilder,
        sampler::{
            DirectSamplerSelection, SamplerBuilder, SamplerHookBuilder, StagedSamplerSelection,
            StepHookBound,
        },
    },
    hooks::{
        on_policy::{LearningRateSchedule, LearningSchedule},
        sampler::EpisodeBoundHook,
    },
};

type DefaultOnPolicyAlgorithm<A, EB, SH> = OnPolicyAlgorithm<
    A,
    DirectSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
    DefaultOnPolicyAlgorithmHooks<
        A,
        DirectSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
        DirectSampler<<EB as EnvBuilder>::Env, EpisodeBoundHook<<EB as EnvBuilder>::Env>>,
    >,
>;

type DefaultOnPolicyAlgorithmFor<AB, EB, SH> =
    DefaultOnPolicyAlgorithm<<AB as AgentBuilder>::Agent, EB, SH>;

type StagedOnPolicyAlgorithm<A, EB, SH> = OnPolicyAlgorithm<
    A,
    StagedSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
    DefaultOnPolicyAlgorithmHooks<
        A,
        StagedSampler<<EB as EnvBuilder>::Env, <SH as SamplerHookBuilder>::Target>,
        DefaultAdapter,
        <EB as EnvBuilder>::Env,
        StagedSampler<<EB as EnvBuilder>::Env, EpisodeBoundHook<<EB as EnvBuilder>::Env>>,
    >,
>;

type StagedOnPolicyAlgorithmFor<AB, EB, SH> =
    StagedOnPolicyAlgorithm<<AB as AgentBuilder>::Agent, EB, SH>;

type DirectDefaultOnPolicyAlgorithmHooks<A, S, C, EB> = DefaultOnPolicyAlgorithmHooks<
    A,
    S,
    C,
    <EB as EnvBuilder>::Env,
    DirectSampler<<EB as EnvBuilder>::Env, EpisodeBoundHook<<EB as EnvBuilder>::Env>>,
>;

type NormalizedDefaultOnPolicyAlgorithmHooks<A, C, EB, H> = DefaultOnPolicyAlgorithmHooks<
    A,
    StagedSampler<<EB as EnvBuilder>::Env, H>,
    C,
    <EB as EnvBuilder>::Env,
    StagedSampler<<EB as EnvBuilder>::Env, EpisodeBoundHook<<EB as EnvBuilder>::Env>>,
>;

/// Internal builder for the default on-policy algorithm lifecycle hooks.
#[derive(Serialize, Deserialize)]
pub(crate) struct DefaultOnPolicyAlgorithmHooksBuilder<EB: EnvBuilder> {
    pub(crate) learning_rate_schedule: Option<LearningRateSchedule>,
    pub(crate) learning_schedule: LearningSchedule,
    pub(crate) evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    #[serde(skip)]
    pub(crate) command_rx: Option<OnPolicyCommandReceiver>,
}

impl<EB: EnvBuilder> Default for DefaultOnPolicyAlgorithmHooksBuilder<EB> {
    fn default() -> Self {
        Self {
            learning_rate_schedule: None,
            learning_schedule: LearningSchedule::rollout_bound(300),
            evaluator_builder: None,
            command_rx: None,
        }
    }
}

impl<EB: EnvBuilder> DefaultOnPolicyAlgorithmHooksBuilder<EB> {
    /// Replaces the learning schedule that controls training termination.
    fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }

    /// Installs or clears the evaluator used during training.
    fn with_evaluator(mut self, evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>) -> Self {
        self.evaluator_builder = evaluator_builder;
        self
    }

    /// Installs the external command receiver used during training.
    fn with_command_rx(mut self, command_rx: OnPolicyCommandReceiver) -> Self {
        self.command_rx = Some(command_rx);
        self
    }

    fn build_direct<A, S, C>(self) -> DirectDefaultOnPolicyAlgorithmHooks<A, S, C, EB>
    where
        A: Agent,
        S: Sampler<Tensor = <EB::Env as Env>::Tensor>,
        C: OnPolicyAdapters<A::Actor, S>,
    {
        let evaluator = self
            .evaluator_builder
            .map(|evaluator_builder| evaluator_builder.build::<A::Actor>());
        DefaultOnPolicyAlgorithmHooks::new(
            self.learning_schedule,
            evaluator,
            self.learning_rate_schedule,
            self.command_rx,
        )
    }

    fn build_normalized<A, C, H>(
        self,
        sampler: &StagedSampler<EB::Env, H>,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
    ) -> NormalizedDefaultOnPolicyAlgorithmHooks<A, C, EB, H>
    where
        A: Agent,
        H: StagedSamplerHook<E = EB::Env>,
        C: OnPolicyAdapters<A::Actor, StagedSampler<EB::Env, H>>,
    {
        let evaluator = self.evaluator_builder.map(|evaluator_builder| {
            let eval_sampler = StagedSampler::build_with_obs_normalizer(
                evaluator_builder.env_builder().clone(),
                EpisodeBoundHook::new(evaluator_builder.n_episodes()),
                evaluator_builder.execution_mode(),
                obs_normalizer,
            );
            evaluator_builder.build_with_sampler::<A::Actor, _>(eval_sampler)
        });
        DefaultOnPolicyAlgorithmHooks::new(
            self.learning_schedule,
            evaluator,
            self.learning_rate_schedule,
            self.command_rx,
        )
    }
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
#[derive(Serialize, Deserialize)]
pub struct OnPolicyAlgorithmBuilder<
    AB: AgentBuilder,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env>,
    ST = DirectSamplerSelection,
> {
    pub(crate) sampler_builder: SamplerBuilder<EB, SH, ST>,
    pub(crate) hooks_builder: DefaultOnPolicyAlgorithmHooksBuilder<EB>,
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
            hooks_builder: DefaultOnPolicyAlgorithmHooksBuilder::default(),
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
            hooks_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(hook_builder),
            agent_builder,
            hooks_builder,
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
            hooks_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_hook(rollout_bound),
            agent_builder,
            hooks_builder,
            seed,
        }
    }

    /// Installs or clears the evaluator used during training.
    pub fn with_evaluator(
        mut self,
        evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    ) -> Self {
        self.hooks_builder = self.hooks_builder.with_evaluator(evaluator_builder);
        self
    }

    /// Replaces the learning schedule that controls training termination.
    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.hooks_builder = self.hooks_builder.with_learning_schedule(learning_schedule);
        self
    }

    /// Installs the external command receiver used during training.
    pub fn with_command_rx(mut self, command_rx: OnPolicyCommandReceiver) -> Self {
        self.hooks_builder = self.hooks_builder.with_command_rx(command_rx);
        self
    }

    /// Sets the learning-rate schedule applied over the training duration.
    pub fn with_learning_rate_schedule(
        mut self,
        learning_rate_schedule: Option<LearningRateSchedule>,
    ) -> Self {
        self.hooks_builder.learning_rate_schedule = learning_rate_schedule;
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
        let evaluator_builder =
            if let Some(evaluator_builder) = self.hooks_builder.evaluator_builder.take() {
                evaluator_builder.with_n_episodes(n_episodes)
            } else {
                let env_builder = self.sampler_builder.env_builder.clone();
                BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
                    .with_n_episodes(n_episodes)
            };
        self.hooks_builder.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Replaces the environment builder used by the evaluator.
    pub fn with_evaluator_env_builder(
        mut self,
        env_builder: r2l_core::env::EnvBuilderType<EB>,
    ) -> Self {
        let evaluator_builder =
            if let Some(evaluator_builder) = self.hooks_builder.evaluator_builder.take() {
                evaluator_builder.with_env_builder(env_builder)
            } else {
                BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
            };
        self.hooks_builder.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets how evaluation environments are executed.
    pub fn with_evaluator_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        let evaluator_builder =
            if let Some(evaluator_builder) = self.hooks_builder.evaluator_builder.take() {
                evaluator_builder.with_execution_mode(execution_mode)
            } else {
                let env_builder = self.sampler_builder.env_builder.clone();
                BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
                    .with_execution_mode(execution_mode)
            };
        self.hooks_builder.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets the filesystem path used to persist the best-performing actor.
    pub fn with_evaluator_best_actor_path<P: Into<std::path::PathBuf>>(
        mut self,
        eval_path: P,
    ) -> Self {
        let evaluator_builder =
            if let Some(evaluator_builder) = self.hooks_builder.evaluator_builder.take() {
                evaluator_builder.with_best_actor_path(eval_path)
            } else {
                let env_builder = self.sampler_builder.env_builder.clone();
                BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
                    .with_best_actor_path(eval_path)
            };
        self.hooks_builder.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets the filesystem path used to persist evaluation states as CSV.
    pub fn with_csv_states<P: Into<std::path::PathBuf>>(mut self, csv_states_path: P) -> Self {
        let evaluator_builder =
            if let Some(evaluator_builder) = self.hooks_builder.evaluator_builder.take() {
                evaluator_builder.with_csv_states(csv_states_path)
            } else {
                let env_builder = self.sampler_builder.env_builder.clone();
                BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
                    .with_csv_states(csv_states_path)
            };
        self.hooks_builder.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets the frequency with which the evaluator runs
    pub fn with_evaluator_frequency(mut self, evaluator_frequency: usize) -> Self {
        assert!(evaluator_frequency > 0);
        let evaluator_builder =
            if let Some(evaluator_builder) = self.hooks_builder.evaluator_builder.take() {
                evaluator_builder.with_evaluator_frequency(evaluator_frequency)
            } else {
                let env_builder = self.sampler_builder.env_builder.clone();
                BestActorEvaluatorBuilder::from_env_builder_type(env_builder)
                    .with_evaluator_frequency(evaluator_frequency)
            };
        self.hooks_builder.evaluator_builder = Some(evaluator_builder);
        self
    }

    /// Sets how training environments are executed.
    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.sampler_builder = self.sampler_builder.with_execution_mode(location);
        self
    }

    /// Switches to staged sampling with an optional observation normalizer.
    pub fn with_observation_normalizer(
        self,
        obs_clip: Option<f32>,
    ) -> OnPolicyAlgorithmBuilder<AB, EB, SH, StagedSamplerSelection> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            hooks_builder,
            agent_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder: sampler_builder.with_obs_normalizer(obs_clip),
            hooks_builder,
            agent_builder,
            seed,
        }
    }
}

impl<AB: AgentBuilder, EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>, ST>
    OnPolicyAlgorithmBuilder<AB, EB, StepHookBound<EB::Env>, ST>
{
    /// Enables reward normalization for step-bounded training rollouts.
    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        self.sampler_builder.hook_builder = self
            .sampler_builder
            .hook_builder
            .with_reward_normalizer(gamma, clip_reward);
        self
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
                DirectSampler<<EB as EnvBuilder>::Env, SH::Target>,
            >,
    {
        if let Some(seed) = self.seed {
            set_seed(seed);
        }
        let env_description = self.sampler_builder.env_builder.env_description()?;
        let sampler = self.sampler_builder.build();
        let observation_size = env_description.observation_size();
        let action_space = env_description.action_space;
        let agent = self
            .agent_builder
            .build(observation_size, action_space, self.seed)?;
        let hooks = self.hooks_builder.build_direct();
        Ok(OnPolicyAlgorithm::new(
            OnPolicyRuntime {
                sampler,
                agent,
                adapter: DefaultAdapter,
            },
            hooks,
        ))
    }
}

impl<
    AB: AgentBuilder,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env, Target: StagedSamplerHook<E = <EB as EnvBuilder>::Env>>,
> OnPolicyAlgorithmBuilder<AB, EB, SH, StagedSamplerSelection>
{
    /// Builds the configured on-policy algorithm runtime using staged sampling.
    pub fn build(self) -> anyhow::Result<StagedOnPolicyAlgorithmFor<AB, EB, SH>>
    where
        DefaultAdapter: OnPolicyAdapters<
                <<AB as AgentBuilder>::Agent as Agent>::Actor,
                StagedSampler<<EB as EnvBuilder>::Env, SH::Target>,
            >,
    {
        if let Some(seed) = self.seed {
            set_seed(seed);
        }
        let env_description = self.sampler_builder.env_builder.env_description()?;
        let observation_size = env_description.observation_size();
        let action_space = env_description.action_space;
        let sampler = self.sampler_builder.build();
        let agent = self
            .agent_builder
            .build(observation_size, action_space, self.seed)?;
        let hook_normalizer = sampler
            .core
            .obs_normalizer
            .as_ref()
            .map(|n| ClippedNormalizer {
                normalizer_mode: NormalizerMode::ReadOnly,
                inner: n.inner.clone(),
            });
        let hooks = self
            .hooks_builder
            .build_normalized(&sampler, hook_normalizer);
        Ok(OnPolicyAlgorithm::new(
            OnPolicyRuntime {
                sampler,
                agent,
                adapter: DefaultAdapter,
            },
            hooks,
        ))
    }
}
