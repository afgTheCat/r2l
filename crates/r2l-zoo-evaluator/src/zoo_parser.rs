use std::{collections::BTreeMap, fs, path::PathBuf};

use r2l_api::{
    BurnBackend, DefaultOnPolicyAlgorithmHooks, EpisodeBoundHook, LearningSchedule,
    PPOAlgorithmBuilder, PPOBurnAgent, StepBoundHook, StepHookBound,
};
use r2l_core::on_policy::algorithm::{DefaultAdapter, OnPolicyAlgorithm};
use r2l_gym::GymEnv;
use r2l_sampler::R2lNormalizedSampler;
use serde::{Deserialize, Deserializer, Serialize, de};
use yaml_serde::Value;

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
    Options { norm_obs: bool, norm_reward: bool },
}

impl RlZooNormalize {
    fn norm_obs(&self) -> bool {
        match self {
            Self::Enabled(enabled) => *enabled,
            Self::Options { norm_obs, .. } => *norm_obs,
        }
    }

    fn norm_reward(&self) -> bool {
        match self {
            Self::Enabled(enabled) => *enabled,
            Self::Options { norm_reward, .. } => *norm_reward,
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
        let value = yaml_serde::from_value::<String>(value)
            .map_err(|err| de::Error::custom(err.to_string()))?;
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

#[derive(Debug, Serialize, Deserialize)]
pub struct RlZooEnvironmentConfig {
    n_envs: usize,
    n_timesteps: usize,
    policy: String,
    n_steps: usize,
    batch_size: usize,
    gae_lambda: f32,
    gamma: f32,
    n_epochs: usize,
    ent_coef: f32,
    learning_rate: RlZooSchedule,
    clip_range: RlZooSchedule,
    vf_coef: f32,
    max_grad_norm: f32,
    normalize: RlZooNormalize,
    use_sde: bool,
    sde_sample_freq: i32,
}

impl RlZooEnvironmentConfig {
    fn suppoerted(&self) -> bool {
        self.policy == "MlpPolicy"
    }

    pub fn build_burn_ppo_algorithm(
        &self,
        env_name: &str,
        csv_path: PathBuf,
    ) -> anyhow::Result<RlZooPpoAlgorithm> {
        let obs_clip = self.normalize.norm_obs().then_some(10.0);
        let mut builder = PPOAlgorithmBuilder::gym(env_name, self.n_envs)
            .with_burn()
            .with_rollout_bound(StepHookBound::new(self.n_steps))
            .with_learning_schedule(LearningSchedule::total_step_bound(self.n_timesteps))
            .with_csv_states(csv_path)
            .with_observation_normalizer(obs_clip)
            .with_lambda(self.gae_lambda)
            .with_gamma(self.gamma)
            .with_total_epochs(self.n_epochs)
            .with_entropy_coeff(self.ent_coef)
            .with_sample_size(self.batch_size)
            .with_learning_rate(self.learning_rate.initial_value())
            .with_clip_range(self.clip_range.initial_value() as f32)
            .with_vf_coeff(Some(self.vf_coef))
            .with_gradient_clipping(Some(self.max_grad_norm));
        if self.normalize.norm_reward() {
            builder = builder.with_reward_normalizer(self.gamma, 10.0);
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
        parsed_content.remove("atari");
        let mut supported_envs = BTreeMap::new();
        let mut unsupported_envs = BTreeMap::new();
        for (env_name, val) in parsed_content {
            let rl_zoo_config = yaml_serde::from_value::<RlZooEnvironmentConfig>(val).unwrap();
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
