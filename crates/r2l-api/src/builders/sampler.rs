use std::marker::PhantomData;

use r2l_core::{
    env::{
        Env, EnvBuilder, EnvBuilderType, EnvDescription,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    on_policy::algorithm::Sampler,
    tensor::R2lTensor,
};
use r2l_sampler::{
    DirectSampler, DirectSamplerHook, SamplerExecutionMode, StagedSampler, StagedSamplerHook,
};
use serde::{Deserialize, Serialize};

use crate::{
    InferenceObservationMode,
    hooks::sampler::{EpisodeBoundHook, StepBoundHook},
    utils::RewardNormalizer,
};

/// Builder trait for sampler hook configurations.
///
/// Implementations of this trait package a rollout-collection policy into a
/// type that can later construct the concrete hook consumed by
/// [`DirectSampler`]. This is the sampler equivalent of choosing a rollout
/// bound in the original sampler interface, but generalized to a hook-driven
/// collection model.
pub trait SamplerHookBuilder {
    /// Environment type collected by the resulting hook.
    type Env: Env;
    /// Concrete sampler hook produced by this builder.
    type Target: DirectSamplerHook<E = Self::Env>;

    /// Builds the hook used by [`ConfiguredSamplerBuilder`] when constructing a sampler.
    fn build(self, n_envs: usize) -> Self::Target;
}

/// Result of constructing a sampler and its shared observation normalization state.
pub struct BuiltSampler<S: Sampler> {
    /// Constructed rollout sampler.
    pub sampler: S,
    /// Read-only observation normalizer shared with evaluation, when configured.
    pub obs_normalizer: Option<ClippedNormalizer<S::Tensor>>,
}

/// Builds a sampler while hiding its concrete environment, hook, and storage configuration.
pub trait SamplerBuilder {
    /// Environment builder used by the sampler.
    type EnvBuilder: EnvBuilder;
    /// Concrete sampler produced by this builder.
    type Sampler: Sampler<Tensor = <<Self::EnvBuilder as EnvBuilder>::Env as Env>::Tensor>;

    /// Returns the configured environment builders.
    fn env_builder(&self) -> &EnvBuilderType<Self::EnvBuilder>;

    /// Returns a representative environment description.
    fn env_description(
        &self,
    ) -> anyhow::Result<EnvDescription<<Self::Sampler as Sampler>::Tensor>> {
        self.env_builder().env_description()
    }

    /// Returns how observations must be processed during inference.
    fn inference_observation_mode(&self) -> InferenceObservationMode {
        InferenceObservationMode::Raw
    }

    /// Sets where sampler workers execute.
    fn with_execution_mode(self, execution_mode: SamplerExecutionMode) -> Self;

    /// Builds the sampler and any normalization state needed by evaluation.
    fn build(self) -> BuiltSampler<Self::Sampler>;
}

#[derive(Serialize, Deserialize)]
struct RewardNormalizerParams {
    gamma: f32,
    clip_reward: f32,
}

impl RewardNormalizerParams {
    fn build_normalizer(&self, n_envs: usize) -> RewardNormalizer {
        RewardNormalizer::new(n_envs, self.gamma, self.clip_reward)
    }
}

/// Step-bounded sampler hook configuration.
///
/// This hook builder configures rollout collection to stop after a fixed
/// number of environment steps have been collected per active worker.
#[derive(Serialize, Deserialize)]
pub struct StepHookBound<E: Env<Tensor: R2lTensor>> {
    n_step: usize,
    reward_normalizer: Option<RewardNormalizerParams>,
    #[serde(skip)]
    _phantom: PhantomData<E>,
}

impl<E: Env<Tensor: R2lTensor>> StepHookBound<E> {
    /// Creates a step-bounded sampler hook configuration.
    pub fn new(n_step: usize) -> Self {
        Self {
            n_step,
            reward_normalizer: None,
            _phantom: PhantomData,
        }
    }

    /// Enables reward normalization with the given discount factor and clipping bound.
    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        self.reward_normalizer = Some(RewardNormalizerParams { gamma, clip_reward });
        self
    }
}

impl<E: Env<Tensor: R2lTensor>> SamplerHookBuilder for StepHookBound<E> {
    type Env = E;
    type Target = StepBoundHook<Self::Env>;

    fn build(self, n_env: usize) -> Self::Target {
        let reward_normalizer = self.reward_normalizer.map(|p| p.build_normalizer(n_env));
        StepBoundHook::new(self.n_step, reward_normalizer)
    }
}

/// Episode-bounded sampler hook configuration.
///
/// This hook builder configures rollout collection to stop after a fixed
/// number of completed episodes have been collected per active worker.
#[derive(Serialize, Deserialize)]
pub struct EpisodeHookBound<E: Env> {
    n_episodes: usize,
    _phantom: PhantomData<E>,
}

impl<E: Env> EpisodeHookBound<E> {
    /// Creates an episode-bounded sampler hook configuration.
    pub fn new(n_episodes: usize) -> Self {
        Self {
            n_episodes,
            _phantom: PhantomData,
        }
    }
}

impl<E: Env> SamplerHookBuilder for EpisodeHookBound<E> {
    type Env = E;
    type Target = EpisodeBoundHook<Self::Env>;

    fn build(self, _n_envs: usize) -> Self::Target {
        EpisodeBoundHook::new(self.n_episodes)
    }
}

/// Marker selecting raw, unnormalized rollout storage.
#[derive(Serialize, Deserialize)]
pub struct DirectSamplerSelection;

/// Marker selecting observation-normalized rollout storage.
#[derive(Serialize, Deserialize)]
pub struct StagedSamplerSelection {
    pub(crate) obs_clip: Option<f32>,
}

/// Configures environment creation, rollout hooks, normalization, and execution.
///
/// [`DefaultSamplerBuilder::new`] creates a homogeneous sampler using `n_envs`
/// copies of one environment builder, a [`StepHookBound`] of `1024`, and
/// [`SamplerExecutionMode::SingleThreaded`].
#[derive(Serialize, Deserialize)]
pub struct ConfiguredSamplerBuilder<
    EB: EnvBuilder,
    S: SamplerHookBuilder<Env = EB::Env>,
    ST = DirectSamplerSelection,
> {
    pub(crate) env_builder: EnvBuilderType<EB>,
    pub(crate) hook_builder: S,
    pub(crate) execution_mode: SamplerExecutionMode,
    pub(crate) sampler_type: ST,
}

/// Default sampler builder using a step-bounded rollout policy.
pub type DefaultSamplerBuilder<EB> =
    ConfiguredSamplerBuilder<EB, StepHookBound<<EB as EnvBuilder>::Env>>;

impl<EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>> DefaultSamplerBuilder<EB> {
    /// Creates a sampler builder from a single environment builder and count.
    ///
    /// The provided builder is replicated into a homogeneous environment set
    /// with `n_envs` copies.
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let env_builder = EnvBuilderType::homogeneous(builder.into(), n_envs);
        Self {
            env_builder,
            hook_builder: StepHookBound::new(1024),
            execution_mode: SamplerExecutionMode::SingleThreaded,
            sampler_type: DirectSamplerSelection,
        }
    }
}

impl<EB: EnvBuilder, S: SamplerHookBuilder<Env = EB::Env>, ST> ConfiguredSamplerBuilder<EB, S, ST> {
    /// Replaces the rollout hook policy used by the sampler.
    ///
    /// This changes the hook-builder type carried by the builder, allowing
    /// callers to swap between the standard step/episode hook bounds or install
    /// a custom sampler hook configuration.
    pub fn with_hook<S2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        hook_builder: S2,
    ) -> ConfiguredSamplerBuilder<EB, S2, ST> {
        let ConfiguredSamplerBuilder {
            env_builder,
            execution_mode,
            sampler_type,
            ..
        } = self;
        ConfiguredSamplerBuilder {
            env_builder,
            execution_mode,
            hook_builder,
            sampler_type,
        }
    }

    /// Sets where the sampler should execute.
    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.execution_mode = location;
        self
    }

    /// Replaces the full environment builder configuration.
    ///
    /// This is useful when you need heterogeneous environments or when the
    /// default homogeneous setup created by [`new`](DefaultSamplerBuilder::new)
    /// is not enough.
    pub fn with_env_builder(mut self, env_builder: EnvBuilderType<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

    /// Switches this builder to normalized sampling with an optional observation normalizer.
    pub fn with_obs_normalizer(
        self,
        obs_clip: Option<f32>,
    ) -> ConfiguredSamplerBuilder<EB, S, StagedSamplerSelection> {
        let ConfiguredSamplerBuilder {
            env_builder,
            hook_builder,
            execution_mode,
            ..
        } = self;
        ConfiguredSamplerBuilder {
            env_builder,
            hook_builder,
            execution_mode,
            sampler_type: StagedSamplerSelection { obs_clip },
        }
    }
}

impl<EB: EnvBuilder, S: SamplerHookBuilder<Env = EB::Env>> SamplerBuilder
    for ConfiguredSamplerBuilder<EB, S, DirectSamplerSelection>
{
    type EnvBuilder = EB;
    type Sampler = DirectSampler<EB::Env, S::Target>;

    fn env_builder(&self) -> &EnvBuilderType<Self::EnvBuilder> {
        &self.env_builder
    }

    fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    /// Builds the configured sampler instance.
    fn build(self) -> BuiltSampler<Self::Sampler> {
        let n_envs = self.env_builder.num_envs();
        let hook = self.hook_builder.build(n_envs);
        BuiltSampler {
            sampler: DirectSampler::build(self.env_builder, hook, self.execution_mode),
            obs_normalizer: None,
        }
    }
}

impl<
    EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>,
    S: SamplerHookBuilder<Env = EB::Env, Target: StagedSamplerHook<E = <EB as EnvBuilder>::Env>>,
> SamplerBuilder for ConfiguredSamplerBuilder<EB, S, StagedSamplerSelection>
{
    type EnvBuilder = EB;
    type Sampler = StagedSampler<EB::Env, S::Target>;

    fn env_builder(&self) -> &EnvBuilderType<Self::EnvBuilder> {
        &self.env_builder
    }

    fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    fn inference_observation_mode(&self) -> InferenceObservationMode {
        if self.sampler_type.obs_clip.is_some() {
            InferenceObservationMode::Normalized
        } else {
            InferenceObservationMode::Raw
        }
    }

    /// Builds the configured staged sampler instance.
    fn build(self) -> BuiltSampler<Self::Sampler> {
        let n_envs = self.env_builder.num_envs();
        let hook = self.hook_builder.build(n_envs);
        let obs_normalizer = self.sampler_type.obs_clip.map(|clip| {
            let env_description = self.env_builder.env_description().unwrap();
            let obs_size = env_description.observation_space.size();
            ClippedNormalizer::build(NormalizerMode::Update, clip, vec![obs_size])
        });
        let sampler = StagedSampler::build_with_obs_normalizer(
            self.env_builder,
            hook,
            self.execution_mode,
            obs_normalizer,
        );
        let obs_normalizer =
            sampler
                .core
                .obs_normalizer
                .as_ref()
                .map(|normalizer| ClippedNormalizer {
                    normalizer_mode: NormalizerMode::ReadOnly,
                    inner: normalizer.inner.clone(),
                });
        BuiltSampler {
            sampler,
            obs_normalizer,
        }
    }
}

#[cfg(test)]
mod test {
    use r2l_gym::GymEnvBuilder;

    use crate::builders::sampler::DefaultSamplerBuilder;

    #[test]
    fn serialize_sampler_builder() {
        let sampler_builder = DefaultSamplerBuilder::<GymEnvBuilder>::new("", 10);
        let serialized = yaml_serde::to_string(&sampler_builder).unwrap();
        println!("{serialized}");
    }
}
