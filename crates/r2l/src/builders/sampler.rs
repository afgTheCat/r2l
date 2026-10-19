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
use r2l_sampler::{RolloutMode, SamplerExecutionMode};
use serde::{Deserialize, Serialize};

use crate::{OnPolicySampler, SamplerHook, utils::RewardNormalizer};

/// Controls observation normalization and optional clipping after normalization.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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

/// Controls reward normalization using discounted returns and clipping.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum RewardNormalizerConfig {
    /// Leave rewards unnormalized.
    #[default]
    Disabled,
    /// Normalize rewards using the running variance of discounted returns.
    Enabled {
        /// Discount factor used to track discounted returns.
        gamma: f32,
        /// Absolute limit applied to normalized rewards.
        clip_reward: f32,
    },
}

#[derive(Serialize, Deserialize)]
pub(crate) struct SamplerConfiguration {
    pub(crate) rollout_mode: RolloutMode,
    pub(crate) execution_mode: SamplerExecutionMode,
    pub(crate) reward_normalizer: RewardNormalizerConfig,
    pub(crate) obs_normalizer: ObsNormalizerConfig,
}

impl Default for SamplerConfiguration {
    fn default() -> Self {
        Self {
            rollout_mode: RolloutMode::step_bound(1024),
            execution_mode: SamplerExecutionMode::MultiThreaded,
            reward_normalizer: RewardNormalizerConfig::Disabled,
            obs_normalizer: ObsNormalizerConfig::Disabled,
        }
    }
}

impl SamplerConfiguration {
    pub(crate) fn build_obs_normalizer<T: R2lTensor>(&self, shape: Shape) -> Option<Normalizer<T>> {
        match self.obs_normalizer {
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
    ) -> Result<OnPolicySampler<E>, Error> {
        let reward_normalizer = match self.reward_normalizer {
            RewardNormalizerConfig::Disabled => None,
            RewardNormalizerConfig::Enabled { gamma, clip_reward } => Some(RewardNormalizer::new(
                NonZeroUsize::new(env_builder.num_envs())
                    .expect("environment builders are nonempty"),
                gamma,
                clip_reward,
            )),
        };
        let hook = SamplerHook::new(self.rollout_mode, reward_normalizer);
        match obs_normalizer {
            Some(normalizer) => {
                OnPolicySampler::build_staged(env_builder, hook, self.execution_mode, normalizer)
            }
            None => OnPolicySampler::build_direct(env_builder, hook, self.execution_mode),
        }
    }
}
