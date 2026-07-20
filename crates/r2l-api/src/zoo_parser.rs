use std::{collections::BTreeMap, fs, path::PathBuf};

use r2l_core::on_policy::algorithm::{DefaultAdapter, OnPolicyAlgorithm};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::R2lNormalizedSampler;
use serde::{Deserialize, Deserializer, Serialize, de};
use yaml_serde::Value;

use crate::{
    BurnBackend, DefaultOnPolicyAlgorithmHooks, EpisodeBoundHook, LearningSchedule,
    PPOAlgorithmBuilder, PPOBurnAgent, PPOBurnAlgorithmBuilder, StepBoundHook, StepHookBound,
    builders::sampler::NormalizedSamplerSelection,
};

pub type RlZooPpoBuilder =
    PPOBurnAlgorithmBuilder<GymEnvBuilder, StepHookBound<GymEnv>, NormalizedSamplerSelection>;
pub type RlZooPpoAlgorithm = OnPolicyAlgorithm<
    PPOBurnAgent<BurnBackend>,
    R2lNormalizedSampler<GymEnv, StepBoundHook<GymEnv>>,
    DefaultOnPolicyAlgorithmHooks<
        PPOBurnAgent<BurnBackend>,
        R2lNormalizedSampler<GymEnv, StepBoundHook<GymEnv>>,
        DefaultAdapter,
        GymEnv,
        R2lNormalizedSampler<GymEnv, EpisodeBoundHook<GymEnv>>,
    >,
>;

#[derive(Debug, Serialize, Deserialize)]
pub struct RlZooDefault {
    n_timesteps: f32,
    policy: String,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum RlZooSchedule {
    Constant(f64),
    Linear(f64),
}

impl RlZooSchedule {
    fn initial_value(self) -> f64 {
        match self {
            Self::Constant(value) | Self::Linear(value) => value,
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

        let value = yaml_serde::from_value::<String>(value)
            .map_err(|err| de::Error::custom(err.to_string()))?;
        if let Some(value) = value.strip_prefix("lin_") {
            let value = value.parse().map_err(de::Error::custom)?;
            return Ok(Self::Linear(value));
        }

        Err(de::Error::custom(format!(
            "unsupported RL Zoo schedule: {value}"
        )))
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum RlZooNormalize {
    Enabled(bool),
    Options {
        norm_obs: Option<bool>,
        norm_reward: Option<bool>,
    },
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
        let value = yaml_serde::from_value::<String>(value)
            .map_err(|err| de::Error::custom(err.to_string()))?;
        let norm_obs = parse_python_bool_option(&value, "norm_obs");
        let norm_reward = parse_python_bool_option(&value, "norm_reward");
        if norm_obs.is_some() || norm_reward.is_some() {
            return Ok(Self::Options {
                norm_obs,
                norm_reward,
            });
        }
        Err(de::Error::custom(format!(
            "unsupported RL Zoo normalize value: {value}"
        )))
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

#[derive(Debug, Serialize, Deserialize)]
pub struct RlZooEnvironmentConfig {
    n_envs: Option<usize>,
    n_timesteps: Option<f32>,
    policy: Option<String>,
    n_steps: Option<usize>,
    batch_size: Option<usize>,
    gae_lambda: Option<f32>,
    gamma: Option<f32>,
    n_epochs: Option<usize>,
    ent_coef: Option<f32>,
    learning_rate: Option<RlZooSchedule>,
    clip_range: Option<RlZooSchedule>,
    vf_coef: Option<f32>,
    max_grad_norm: Option<f32>,
    normalize: Option<RlZooNormalize>,
    use_sde: Option<bool>,
    sde_sample_freq: Option<usize>,
}

impl RlZooEnvironmentConfig {
    fn merge_with_default(mut self, rl_zoo_default: &RlZooDefault) -> Self {
        if self.n_timesteps.is_none() {
            self.n_timesteps = Some(rl_zoo_default.n_timesteps);
        };
        if self.policy.is_none() {
            self.policy = Some(rl_zoo_default.policy.clone());
        }
        self
    }

    fn suppoerted(&self) -> bool {
        if let Some(policy) = &self.policy
            && policy != "MlpPolicy"
        {
            return false;
        }
        true
    }

    pub fn build_burn_ppo_algorithm(&self, env_name: &str) -> anyhow::Result<RlZooPpoAlgorithm> {
        let n_envs = self.n_envs.unwrap();
        let n_steps = self.n_steps.unwrap();
        let n_timesteps = self.n_timesteps.unwrap() as usize;

        let mut builder = PPOAlgorithmBuilder::gym(env_name, n_envs)
            .with_burn()
            .with_rollout_bound(StepHookBound::new(n_steps))
            .with_learning_schedule(LearningSchedule::total_step_bound(n_timesteps))
            .with_observation_normalizer(10.0);

        if let Some(gae_lambda) = self.gae_lambda {
            builder = builder.with_lambda(gae_lambda);
        }
        if let Some(gamma) = self.gamma {
            builder = builder.with_gamma(gamma);
        }
        if let Some(n_epochs) = self.n_epochs {
            builder = builder.with_total_epochs(n_epochs);
        }
        if let Some(ent_coef) = self.ent_coef {
            builder = builder.with_entropy_coeff(ent_coef);
        }
        if let Some(batch_size) = self.batch_size {
            builder = builder.with_sample_size(batch_size);
        }
        if let Some(learning_rate) = self.learning_rate {
            builder = builder.with_learning_rate(learning_rate.initial_value());
        }
        if let Some(clip_range) = self.clip_range {
            builder = builder.with_clip_range(clip_range.initial_value() as f32);
        }
        if let Some(vf_coef) = self.vf_coef {
            builder = builder.with_vf_coeff(Some(vf_coef));
        }
        if let Some(max_grad_norm) = self.max_grad_norm {
            builder = builder.with_gradient_clipping(Some(max_grad_norm));
        }

        builder.build()
    }
}

#[derive(Debug)]
pub struct ZooConfig {
    pub supported_envs: BTreeMap<String, RlZooEnvironmentConfig>,
    pub unsupported_envs: BTreeMap<String, RlZooEnvironmentConfig>,
}

impl ZooConfig {
    pub fn parse_rl_zoo_config(path: PathBuf) -> Self {
        let content = fs::read_to_string(path).unwrap();
        let mut parsed_content: BTreeMap<String, Value> = yaml_serde::from_str(&content).unwrap();
        let rl_zoo_default: RlZooDefault = parsed_content
            .remove("default")
            .map(|val| yaml_serde::from_value(val).unwrap())
            .unwrap();
        // TODO: should we pass this?
        parsed_content.remove("atari");
        let mut supported_envs = BTreeMap::new();
        let mut unsupported_envs = BTreeMap::new();
        for (env_name, val) in parsed_content {
            let rl_zoo_config = yaml_serde::from_value::<RlZooEnvironmentConfig>(val).unwrap();
            let rl_zoo_config = rl_zoo_config.merge_with_default(&rl_zoo_default);
            if rl_zoo_config.suppoerted() {
                supported_envs.insert(env_name, rl_zoo_config);
            } else {
                unsupported_envs.insert(env_name, rl_zoo_config);
            }
        }
        Self {
            supported_envs,
            unsupported_envs,
        }
    }
}
