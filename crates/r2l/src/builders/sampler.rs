use std::num::NonZeroUsize;

use r2l_core::{
    Shape,
    env::{
        Env, EnvBuilderType,
        normalizer::{Normalizer, NormalizerMode},
    },
    error::Error,
    tensor::R2lTensor,
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

pub(crate) struct RewardNormalizerConfig {
    pub(crate) gamma: f32,
    pub(crate) clip_reward: f32,
}

pub(crate) struct SamplerSetup {
    pub(crate) rollout_mode: RolloutMode,
    pub(crate) reward_normalizer: Option<RewardNormalizerConfig>,
    pub(crate) obs_normalizer: ObsNormalizerConfig,
}

pub(crate) struct SamplerConfiguration {
    pub(crate) setup: SamplerSetup,
    pub(crate) execution_mode: SamplerExecutionMode,
}

impl SamplerConfiguration {
    pub(crate) fn build_obs_normalizer<T: R2lTensor>(&self, shape: Shape) -> Option<Normalizer<T>> {
        match self.setup.obs_normalizer {
            ObsNormalizerConfig::Disabled => None,
            ObsNormalizerConfig::Enabled { clip } => {
                Some(Normalizer::build(NormalizerMode::Update, clip, shape))
            }
        }
    }

    pub(crate) fn build_sampler<E: Env + 'static>(
        &self,
        env_builder: &EnvBuilderType<E>,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<RolloutSampler<SamplerHook<E>>, Error> {
        let reward_normalizer = self.setup.reward_normalizer.as_ref().map(|config| {
            RewardNormalizer::new(
                NonZeroUsize::new(env_builder.num_envs())
                    .expect("environment builders are nonempty"),
                config.gamma,
                config.clip_reward,
            )
        });
        let hook = SamplerHook::new(self.setup.rollout_mode, reward_normalizer);
        match obs_normalizer {
            Some(normalizer) => {
                RolloutSampler::build_staged(env_builder, hook, self.execution_mode, normalizer)
            }
            None => RolloutSampler::build_direct(env_builder, hook, self.execution_mode),
        }
    }
}
