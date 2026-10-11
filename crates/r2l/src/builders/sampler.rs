use std::num::NonZeroUsize;

use r2l_core::{
    env::{
        Env, EnvBuilderType,
        normalizer::{Normalizer, NormalizerMode},
    },
    error::Error,
};
use r2l_sampler::{RolloutMode, RolloutSampler, SamplerExecutionMode};

use crate::{
    EpisodeBoundHook, StepBoundHook, hooks::sampler::SamplerHook, utils::RewardNormalizer,
};

/// Controls observation normalization and optional clipping after normalization.
#[derive(Clone, Copy, Debug)]
pub enum ObsNormalizerConfig {
    /// Normalize observations using running mean and variance.
    Enabled {
        /// Absolute limit on normalized values, or `None` to disable clipping.
        clip: Option<f32>,
    },
    /// Leave observations unnormalized.
    Disabled,
}

impl ObsNormalizerConfig {
    /// Sets clipping for enabled normalization; leaves disabled normalization disabled.
    ///
    /// # Arguments
    ///
    /// * `clip` - Absolute limit on normalized observations, or `None` for no clipping.
    #[must_use]
    pub fn with_clip(self, clip: Option<f32>) -> Self {
        match self {
            Self::Disabled => Self::Disabled,
            Self::Enabled { .. } => Self::Enabled { clip },
        }
    }
}

pub(crate) enum SamplerSetup<E: Env> {
    Episode {
        rollout_episodes: NonZeroUsize,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    },
    Step {
        rollout_steps: NonZeroUsize,
        reward_normalizer: Option<RewardNormalizer>,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    },
}

pub(crate) struct SamplerConfiguration<E: Env + 'static> {
    pub(crate) setup: SamplerSetup<E>,
    pub(crate) execution_mode: SamplerExecutionMode,
    pub(crate) env_builder: EnvBuilderType<E>,
}

impl<E: Env> SamplerConfiguration<E> {
    pub(crate) fn n_envs(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.env_builder.num_envs()).expect("environment builders are nonempty")
    }

    pub(crate) fn set_reward_normalizer(&mut self, gamma: f32, clip_reward: f32) {
        let n_envs = self.n_envs();
        let SamplerSetup::Step {
            reward_normalizer, ..
        } = &mut self.setup
        else {
            unreachable!("reward normalization requires step-bound sampling")
        };
        *reward_normalizer = Some(RewardNormalizer::new(n_envs, gamma, clip_reward));
    }

    pub(crate) fn with_observation_normalizer(mut self, config: ObsNormalizerConfig) -> Self {
        let normalizer = match config {
            ObsNormalizerConfig::Disabled => None,
            ObsNormalizerConfig::Enabled { clip } => {
                let description = self.env_builder.env_description().unwrap();
                let shape = description.observation_space.shape().unwrap();
                Some(Normalizer::build(NormalizerMode::Update, clip, shape))
            }
        };
        match &mut self.setup {
            SamplerSetup::Step { obs_normalizer, .. }
            | SamplerSetup::Episode { obs_normalizer, .. } => *obs_normalizer = normalizer,
        }
        self
    }

    pub(crate) fn rollout_mode(&self) -> RolloutMode {
        match &self.setup {
            SamplerSetup::Step { rollout_steps, .. } => RolloutMode::StepBound {
                n_steps: *rollout_steps,
            },
            SamplerSetup::Episode {
                rollout_episodes, ..
            } => RolloutMode::EpisodeBound {
                n_episodes: *rollout_episodes,
            },
        }
    }

    pub(crate) fn obs_normalizer(&self) -> Option<Normalizer<E::Tensor>> {
        match &self.setup {
            SamplerSetup::Step { obs_normalizer, .. }
            | SamplerSetup::Episode { obs_normalizer, .. } => obs_normalizer.clone(),
        }
    }

    fn build_sampler<Bound>(
        &self,
        hook: SamplerHook<E, Bound>,
    ) -> Result<RolloutSampler<E, SamplerHook<E, Bound>>, Error> {
        match self.obs_normalizer() {
            Some(normalizer) => RolloutSampler::build_staged(
                &self.env_builder,
                hook,
                self.execution_mode,
                normalizer.with_mode(NormalizerMode::Update),
            ),
            None => RolloutSampler::build_direct(&self.env_builder, hook, self.execution_mode),
        }
    }

    pub(crate) fn step_bound_sampler(&self) -> Result<RolloutSampler<E, StepBoundHook<E>>, Error> {
        let SamplerSetup::Step {
            reward_normalizer, ..
        } = &self.setup
        else {
            unreachable!("step-bound sampler type must use matching configuration")
        };
        self.build_sampler(StepBoundHook::new(
            self.rollout_mode(),
            reward_normalizer.clone(),
        ))
    }

    pub(crate) fn episode_bound_sampler(
        &self,
    ) -> Result<RolloutSampler<E, EpisodeBoundHook<E>>, Error> {
        let SamplerSetup::Episode { .. } = &self.setup else {
            unreachable!("episode-bound sampler type must use matching configuration")
        };
        self.build_sampler(EpisodeBoundHook::new(self.rollout_mode(), None))
    }
}
