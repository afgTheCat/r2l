use std::num::NonZeroUsize;

use r2l_core::{
    env::{
        Env,
        normalizer::{Normalizer, NormalizerMode},
    },
    error::Error,
};
use r2l_sampler::{DirectSampler, RolloutMode, SamplerExecutionMode, StagedSampler};

use super::environment::EnvBuildPlan;
use crate::{
    EpisodeBoundHook, StepBoundHook, hooks::progress::SharedTrainingProgress,
    utils::RewardNormalizer,
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
    DirectStep {
        rollout_steps: NonZeroUsize,
        reward_normalizer: Option<RewardNormalizer>,
    },
    DirectEpisode {
        rollout_episodes: NonZeroUsize,
    },
    StagedStep {
        rollout_steps: NonZeroUsize,
        reward_normalizer: Option<RewardNormalizer>,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    },
}

pub(crate) struct SamplerConfiguration<E: Env> {
    pub(crate) setup: SamplerSetup<E>,
    pub(crate) execution_mode: SamplerExecutionMode,
    pub(crate) env_build_plan: Box<dyn EnvBuildPlan<E>>,
}

impl<E: Env> SamplerConfiguration<E> {
    pub(crate) fn set_reward_normalizer(&mut self, gamma: f32, clip_reward: f32) {
        let reward_normalizer = match &mut self.setup {
            SamplerSetup::DirectStep {
                reward_normalizer, ..
            }
            | SamplerSetup::StagedStep {
                reward_normalizer, ..
            } => reward_normalizer,
            SamplerSetup::DirectEpisode { .. } => {
                unreachable!("reward normalization requires step-bound sampling")
            }
        };
        *reward_normalizer = Some(RewardNormalizer::new(
            self.env_build_plan.n_envs(),
            gamma,
            clip_reward,
        ));
    }

    pub(crate) fn with_observation_normalizer(mut self, config: ObsNormalizerConfig) -> Self {
        let SamplerSetup::DirectStep {
            rollout_steps,
            reward_normalizer,
        } = self.setup
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        let obs_normalizer = match config {
            ObsNormalizerConfig::Disabled => None,
            ObsNormalizerConfig::Enabled { clip } => {
                let description = self.env_build_plan.env_description();
                let shape = description.observation_space.shape().unwrap();
                Some(Normalizer::build(NormalizerMode::Update, clip, shape))
            }
        };
        self.setup = SamplerSetup::StagedStep {
            rollout_steps,
            reward_normalizer,
            obs_normalizer,
        };
        self
    }

    pub(crate) fn rollout_mode(&self) -> RolloutMode {
        match &self.setup {
            SamplerSetup::DirectStep { rollout_steps, .. }
            | SamplerSetup::StagedStep { rollout_steps, .. } => RolloutMode::StepBound {
                n_steps: *rollout_steps,
            },
            SamplerSetup::DirectEpisode { rollout_episodes } => RolloutMode::EpisodeBound {
                n_episodes: *rollout_episodes,
            },
        }
    }

    pub(crate) fn obs_normalizer(&self) -> Option<Normalizer<E::Tensor>> {
        match &self.setup {
            SamplerSetup::StagedStep { obs_normalizer, .. } => obs_normalizer.clone(),
            _ => None,
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn direct_sampler_step_bound(
        &self,
        progress: SharedTrainingProgress,
    ) -> Result<DirectSampler<E, StepBoundHook<E>>, Error> {
        let SamplerSetup::DirectStep {
            reward_normalizer, ..
        } = &self.setup
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.execution_mode);
        let step_bound_hook = StepBoundHook::new(progress, reward_normalizer.clone());
        Ok(DirectSampler::new(sampler_core, step_bound_hook))
    }

    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn direct_sampler_episode_bound(
        &self,
        progress: SharedTrainingProgress,
    ) -> Result<DirectSampler<E, EpisodeBoundHook<E>>, Error> {
        let SamplerSetup::DirectEpisode { .. } = &self.setup else {
            unreachable!("direct episode-bound sampler type must use matching configuration")
        };
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.execution_mode);
        let episode_bound_hook = EpisodeBoundHook::new(progress, None);
        Ok(DirectSampler::new(sampler_core, episode_bound_hook))
    }

    pub(crate) fn staged_sampler_step_bound(
        &self,
        progress: SharedTrainingProgress,
    ) -> Result<StagedSampler<E, StepBoundHook<E>>, Error> {
        let SamplerSetup::StagedStep {
            reward_normalizer,
            obs_normalizer,
            ..
        } = &self.setup
        else {
            unreachable!("staged step-bound sampler type must use matching configuration")
        };
        let obs_normalizer = obs_normalizer
            .as_ref()
            .map(|normalizer| normalizer.with_mode(NormalizerMode::Update));
        let sampler_core = self
            .env_build_plan
            .build_staged_sampler_core(self.execution_mode, obs_normalizer)?;
        let step_bound_hook = StepBoundHook::new(progress, reward_normalizer.clone());
        Ok(StagedSampler::new(sampler_core, step_bound_hook))
    }
}
