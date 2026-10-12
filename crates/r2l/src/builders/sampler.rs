use std::num::NonZeroUsize;

use r2l_core::{
    env::{
        Env, EnvBuilderType,
        normalizer::{Normalizer, NormalizerMode},
    },
    error::Error,
};
use r2l_sampler::{RolloutMode, RolloutSampler, SamplerExecutionMode};

use crate::{hooks::sampler::SamplerHook, utils::RewardNormalizer};

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

pub(crate) struct SamplerSetup<E: Env> {
    pub(crate) rollout_mode: RolloutMode,
    pub(crate) reward_normalizer: Option<RewardNormalizer>,
    pub(crate) obs_normalizer: Option<Normalizer<E::Tensor>>,
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
        self.setup.reward_normalizer =
            Some(RewardNormalizer::new(self.n_envs(), gamma, clip_reward));
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
        self.setup.obs_normalizer = normalizer;
        self
    }

    pub(crate) fn rollout_mode(&self) -> RolloutMode {
        self.setup.rollout_mode
    }

    pub(crate) fn obs_normalizer(&self) -> Option<Normalizer<E::Tensor>> {
        self.setup.obs_normalizer.clone()
    }

    pub(crate) fn build_sampler(&self) -> Result<RolloutSampler<E, SamplerHook<E>>, Error> {
        let hook = SamplerHook::new(self.rollout_mode(), self.setup.reward_normalizer.clone());
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
}
