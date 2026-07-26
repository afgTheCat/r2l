use std::marker::PhantomData;

use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    tensor::R2lTensor,
};
use r2l_sampler::{
    DirectSampler, DirectSamplerHook, NormalizerMode, SamplerExecutionMode, StagedSampler,
    StagedSamplerHook,
};
use serde::{Deserialize, Serialize};

use crate::{
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

    /// Builds the hook used by [`SamplerBuilder`] when constructing a sampler.
    fn build(self, n_envs: usize) -> Self::Target;
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
pub struct SamplerBuilder<
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
pub type DefaultSamplerBuilder<EB> = SamplerBuilder<EB, StepHookBound<<EB as EnvBuilder>::Env>>;

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

impl<EB: EnvBuilder, S: SamplerHookBuilder<Env = EB::Env>, ST> SamplerBuilder<EB, S, ST> {
    /// Replaces the rollout hook policy used by the sampler.
    ///
    /// This changes the hook-builder type carried by the builder, allowing
    /// callers to swap between the standard step/episode hook bounds or install
    /// a custom sampler hook configuration.
    pub fn with_hook<S2: SamplerHookBuilder<Env = EB::Env>>(
        self,
        hook_builder: S2,
    ) -> SamplerBuilder<EB, S2, ST> {
        let SamplerBuilder {
            env_builder,
            execution_mode,
            sampler_type,
            ..
        } = self;
        SamplerBuilder {
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
    ) -> SamplerBuilder<EB, S, StagedSamplerSelection> {
        let SamplerBuilder {
            env_builder,
            hook_builder,
            execution_mode,
            ..
        } = self;
        SamplerBuilder {
            env_builder,
            hook_builder,
            execution_mode,
            sampler_type: StagedSamplerSelection { obs_clip },
        }
    }
}

impl<EB: EnvBuilder, S: SamplerHookBuilder<Env = EB::Env>>
    SamplerBuilder<EB, S, DirectSamplerSelection>
{
    /// Builds the configured sampler instance.
    pub fn build(self) -> DirectSampler<EB::Env, S::Target> {
        let n_envs = self.env_builder.num_envs();
        let hook = self.hook_builder.build(n_envs);
        DirectSampler::build(self.env_builder, hook, self.execution_mode)
    }
}

impl<
    EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>,
    S: SamplerHookBuilder<Env = EB::Env, Target: StagedSamplerHook<E = <EB as EnvBuilder>::Env>>,
> SamplerBuilder<EB, S, StagedSamplerSelection>
{
    /// Builds the configured normalized sampler instance.
    pub fn build(self) -> StagedSampler<EB::Env, S::Target> {
        let n_envs = self.env_builder.num_envs();
        let hook = self.hook_builder.build(n_envs);
        StagedSampler::build(
            self.env_builder,
            hook,
            self.execution_mode,
            self.sampler_type.obs_clip,
            NormalizerMode::Update,
            false,
        )
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
