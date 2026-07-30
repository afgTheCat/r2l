use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    ActorWrapper,
    env::{Env, EnvBuilder, Snapshot, normalizer::ClippedNormalizer},
    models::Actor,
    rng::sample_u64,
};
use serde::{Deserialize, Serialize};

use crate::{
    CandleBackend, OnPolicyAgentBuilder, PolicyBuilder,
    builders::{
        agent::AgentBuilder,
        normalizer::NormalizerBuilder,
        on_policy::OnPolicyAlgorithmBuilder,
        sampler::{
            ConfiguredSamplerBuilder, DirectSamplerSelection, SamplerBuilder, SamplerHookBuilder,
            StagedSamplerSelection,
        },
    },
};

/// Stateful, single-environment inference runtime.
pub struct Inference2<E: Env, A: Actor<Tensor = E::Tensor>> {
    env: E,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    actor: A,
    last_state: E::Tensor,
}

impl<E: Env, A: Actor<Tensor = E::Tensor>> Inference2<E, A> {
    fn new(
        mut env: E,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
        actor: A,
    ) -> anyhow::Result<Self> {
        let mut last_state = env.reset(sample_u64())?;
        if let Some(obs_normalizer) = &obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut last_state);
        }
        Ok(Self {
            env,
            obs_normalizer,
            actor,
            last_state,
        })
    }

    /// Resets the environment and its current actor observation.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        let mut last_state = self.env.reset(sample_u64())?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut last_state);
        }
        self.last_state = last_state;
        Ok(())
    }

    /// Chooses an action and advances the environment by one step.
    pub fn step(&mut self) -> anyhow::Result<Snapshot<E::Tensor>> {
        let action = self.actor.action(self.last_state.clone())?;
        let mut snapshot = self.env.step(action)?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut snapshot.state);
        }
        self.last_state = snapshot.state.clone();
        Ok(snapshot)
    }
}

/// Raw inference strategy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DirectInferenceBuilder;

/// Observation-normalized inference strategy.
///
/// The learned normalizer is stored separately from the inference
/// configuration and supplied when the runtime is built.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StagedInferenceBuilder;

/// Serializable configuration used to reconstruct an inference runtime.
///
/// `Strategy` records whether inference uses raw or normalized observations.
/// Learned state, such as policy weights and normalizer statistics, is kept in
/// separate artifacts alongside this configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inference2Builder<Strategy = DirectInferenceBuilder, Backend = CandleBackend> {
    policy_builder: PolicyBuilder,
    strategy: Strategy,
    backend: Backend,
}

/// Candle-backed inference runtime.
pub type CandleInference2<E> = Inference2<E, ActorWrapper<CandlePolicyKind, <E as Env>::Tensor>>;

impl<Strategy, Backend> Inference2Builder<Strategy, Backend> {
    /// Creates an inference builder from its concrete configuration parts.
    pub fn new(policy_builder: PolicyBuilder, strategy: Strategy, backend: Backend) -> Self {
        Self {
            policy_builder,
            strategy,
            backend,
        }
    }
}

impl<Strategy> Inference2Builder<Strategy, CandleBackend> {
    fn build_candle<E: Env>(
        self,
        env: E,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> anyhow::Result<CandleInference2<E>> {
        let Self {
            policy_builder,
            strategy: _,
            backend,
        } = self;
        let env_description = env.env_description();
        let actor = policy_builder.build_candle(
            env_description.observation_space.size(),
            env_description.action_space,
            &backend.device,
        )?;
        // TODO: load actor weights from the inference artifact directory.
        Inference2::new(env, obs_normalizer, ActorWrapper::new(actor))
    }
}

impl Inference2Builder<DirectInferenceBuilder, CandleBackend> {
    /// Builds an inference runtime that consumes raw environment observations.
    pub fn build<E: Env>(self, env: E) -> anyhow::Result<CandleInference2<E>> {
        self.build_candle(env, None)
    }
}

impl Inference2Builder<StagedInferenceBuilder, CandleBackend> {
    /// Builds an inference runtime using separately loaded normalizer state.
    pub fn build<E: Env>(
        self,
        env: E,
        normalizer_builder: NormalizerBuilder,
    ) -> anyhow::Result<CandleInference2<E>> {
        self.build_candle(env, Some(normalizer_builder.into_normalizer()))
    }
}

impl<P, H, SB> OnPolicyAlgorithmBuilder<OnPolicyAgentBuilder<P, H, CandleBackend>, SB>
where
    OnPolicyAgentBuilder<P, H, CandleBackend>: AgentBuilder,
    SB: SamplerBuilder,
{
    fn inference_builder_with_strategy<Strategy>(
        &self,
        strategy: Strategy,
    ) -> Inference2Builder<Strategy, CandleBackend> {
        let policy_builder = self
            .agent_builder
            .learning_module_builder
            .policy_builder
            .clone();
        Inference2Builder::new(policy_builder, strategy, self.agent_builder.backend.clone())
    }
}

impl<P, H, EB, SH>
    OnPolicyAlgorithmBuilder<
        OnPolicyAgentBuilder<P, H, CandleBackend>,
        ConfiguredSamplerBuilder<EB, SH, DirectSamplerSelection>,
    >
where
    OnPolicyAgentBuilder<P, H, CandleBackend>: AgentBuilder,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env>,
    ConfiguredSamplerBuilder<EB, SH, DirectSamplerSelection>: SamplerBuilder<EnvBuilder = EB>,
{
    /// Derives the raw-observation inference configuration used by this run.
    pub fn inference_builder(&self) -> Inference2Builder<DirectInferenceBuilder, CandleBackend> {
        self.inference_builder_with_strategy(DirectInferenceBuilder)
    }
}

impl<P, H, EB, SH>
    OnPolicyAlgorithmBuilder<
        OnPolicyAgentBuilder<P, H, CandleBackend>,
        ConfiguredSamplerBuilder<EB, SH, StagedSamplerSelection>,
    >
where
    OnPolicyAgentBuilder<P, H, CandleBackend>: AgentBuilder,
    EB: EnvBuilder,
    SH: SamplerHookBuilder<Env = EB::Env>,
    ConfiguredSamplerBuilder<EB, SH, StagedSamplerSelection>: SamplerBuilder<EnvBuilder = EB>,
{
    /// Derives the normalized-observation inference configuration used by this run.
    pub fn inference_builder(&self) -> Inference2Builder<StagedInferenceBuilder, CandleBackend> {
        self.inference_builder_with_strategy(StagedInferenceBuilder)
    }
}
