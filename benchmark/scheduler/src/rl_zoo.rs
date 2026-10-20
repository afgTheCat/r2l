//! Imports RL Zoo configuration into r2l training settings.

use std::{num::NonZeroUsize, path::Path};

use anyhow::{Context, ensure};
use r2l::{
    AdamWConfig, BackendConfig, ClipRangeSchedule, GradientClippingConfig, LearningRateSchedule,
    ObsNormalizerConfig, OnPolicyConfig, OptimizerConfig, PPOConfig, RewardNormalizerConfig,
    SdeConfig, TrainingArtifactsConfig, TrainingLimit,
};
use r2l_benchmark_task::R2lTask;
use serde::{Deserialize, Deserializer, de};
use yaml_serde::Value;

#[derive(Debug, Clone, Copy)]
pub enum RlZooSchedule {
    Constant(f64),
    Linear(f64),
}

impl RlZooSchedule {
    /// Converts this schedule into a learning-rate schedule.
    #[must_use]
    pub fn into_learning_rate_schedule(self) -> LearningRateSchedule {
        match self {
            Self::Constant(value) => LearningRateSchedule::Constant(value),
            Self::Linear(value) => LearningRateSchedule::Linear(value),
        }
    }

    /// Converts this schedule into a PPO clipping-range schedule.
    #[must_use]
    pub fn into_clip_range_schedule(self) -> ClipRangeSchedule {
        match self {
            Self::Constant(value) => ClipRangeSchedule::Constant(value as f32),
            Self::Linear(value) => ClipRangeSchedule::Linear(value as f32),
        }
    }
}

impl<'de> Deserialize<'de> for RlZooSchedule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if let Ok(value) = yaml_serde::from_value::<f64>(value.clone()) {
            return Ok(Self::Constant(value));
        }

        let value = yaml_serde::from_value::<String>(value).map_err(de::Error::custom)?;
        if let Some(value) = value.strip_prefix("lin_") {
            let value = value.parse().map_err(de::Error::custom)?;
            return Ok(Self::Linear(value));
        }

        Err(de::Error::custom(format!(
            "unsupported RL Zoo schedule: {value}"
        )))
    }
}

#[derive(Debug, Clone)]
pub enum RlZooNormalize {
    Enabled(bool),
    Options { norm_obs: bool, norm_reward: bool },
}

impl RlZooNormalize {
    #[must_use]
    pub fn norm_obs(&self) -> bool {
        match self {
            Self::Enabled(enabled) => *enabled,
            Self::Options { norm_obs, .. } => *norm_obs,
        }
    }

    #[must_use]
    pub fn norm_reward(&self) -> bool {
        match self {
            Self::Enabled(enabled) => *enabled,
            Self::Options { norm_reward, .. } => *norm_reward,
        }
    }

    /// Builds observation normalization settings with clipping disabled.
    #[must_use]
    pub fn to_normalizer_config(&self) -> ObsNormalizerConfig {
        if self.norm_obs() {
            ObsNormalizerConfig::Enabled { clip: None }
        } else {
            ObsNormalizerConfig::Disabled
        }
    }
}

impl<'de> Deserialize<'de> for RlZooNormalize {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if let Ok(enabled) = yaml_serde::from_value::<bool>(value.clone()) {
            return Ok(Self::Enabled(enabled));
        }
        let value = yaml_serde::from_value::<String>(value).map_err(de::Error::custom)?;
        let norm_obs = parse_python_bool_option(&value, "norm_obs").ok_or_else(|| {
            de::Error::custom(format!(
                "missing norm_obs in RL Zoo normalize value: {value}"
            ))
        })?;
        let norm_reward = parse_python_bool_option(&value, "norm_reward").ok_or_else(|| {
            de::Error::custom(format!(
                "missing norm_reward in RL Zoo normalize value: {value}"
            ))
        })?;
        Ok(Self::Options {
            norm_obs,
            norm_reward,
        })
    }
}

fn parse_python_bool_option(value: &str, key: &str) -> Option<bool> {
    let key_pos = value.find(key)?;
    let rest = &value[key_pos + key.len()..];
    let colon_pos = rest.find(':')?;
    let rest = rest[colon_pos + 1..].trim_start();
    if rest.starts_with("True") {
        Some(true)
    } else if rest.starts_with("False") {
        Some(false)
    } else {
        None
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct RlZooEnvironmentConfig {
    pub n_envs: usize,
    pub n_timesteps: usize,
    pub policy: String,
    pub n_steps: usize,
    pub batch_size: usize,
    pub gae_lambda: f32,
    pub gamma: f32,
    pub n_epochs: usize,
    pub ent_coef: f32,
    pub learning_rate: RlZooSchedule,
    pub clip_range: RlZooSchedule,
    pub vf_coef: f32,
    pub max_grad_norm: f32,
    pub log_std_init: f32,
    pub normalize: RlZooNormalize,
    pub use_sde: bool,
    pub sde_sample_freq: i32,
}

impl RlZooEnvironmentConfig {
    #[must_use]
    pub fn supported(&self) -> bool {
        self.policy == "MlpPolicy"
    }

    pub fn to_task(
        &self,
        backend: BackendConfig,
        env_name: String,
        output_dir: &Path,
    ) -> anyhow::Result<R2lTask> {
        let n_envs = NonZeroUsize::new(self.n_envs).context("n_envs must be positive")?;
        let n_steps = NonZeroUsize::new(self.n_steps).context("n_steps must be positive")?;
        let n_timesteps =
            NonZeroUsize::new(self.n_timesteps).context("n_timesteps must be positive")?;
        let sample_size =
            NonZeroUsize::new(self.batch_size).context("batch_size must be positive")?;
        let total_epochs = NonZeroUsize::new(self.n_epochs).context("n_epochs must be positive")?;
        let mut config = OnPolicyConfig {
            seed: Some(0),
            ..Default::default()
        };
        config.algo_config.gamma = self.gamma;
        config.algo_config.lambda = self.gae_lambda;
        config.algo_config.sample_size = sample_size;
        config.hook_config.training_limit = TrainingLimit::TotalStepBound {
            total_steps: n_timesteps,
        };
        config.hook_config.training_artifacts = Some(TrainingArtifactsConfig::new(output_dir));
        config.learner_builder.optimizer = OptimizerConfig::Joint(AdamWConfig {
            learning_rate: self.learning_rate.into_learning_rate_schedule(),
            gradient_clipping: GradientClippingConfig::Norm(self.max_grad_norm),
            ..Default::default()
        });
        config.learner_builder.policy_config.log_std_init = self.log_std_init;
        config.learner_builder.policy_config.sde = if self.use_sde {
            ensure!(
                self.sde_sample_freq >= -1,
                "sde_sample_freq must be -1, zero, or a positive number of steps"
            );
            SdeConfig::Enabled {
                resample_every: usize::try_from(self.sde_sample_freq)
                    .ok()
                    .and_then(NonZeroUsize::new),
            }
        } else {
            SdeConfig::Disabled
        };
        config.learning_hook.entropy_coeff = self.ent_coef;
        config.learning_hook.vf_coeff = self.vf_coef;
        config.sampler_configuration.rollout_mode = r2l::RolloutMode::StepBound { n_steps };
        config.sampler_configuration.obs_normalizer =
            self.normalize.to_normalizer_config().with_clip(Some(10.0));
        config.sampler_configuration.reward_normalizer = if self.normalize.norm_reward() {
            RewardNormalizerConfig::Enabled {
                gamma: self.gamma,
                clip_reward: 10.0,
            }
        } else {
            RewardNormalizerConfig::Disabled
        };
        Ok(R2lTask {
            backend,
            ppo: PPOConfig {
                total_epochs,
                clip_range_schedule: self.clip_range.into_clip_range_schedule(),
                ..Default::default()
            },
            config,
            env_name,
            n_envs,
        })
    }
}
