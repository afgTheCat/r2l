//! Task definitions shared by the benchmark scheduler and worker.

use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use yaml_serde::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Backend {
    Burn,
    Candle,
    Sb3,
}

impl Backend {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Burn => "burn",
            Self::Candle => "candle",
            Self::Sb3 => "sb3",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    pub backend: Backend,
    pub rl_zoo_env_config: RlZooEnvironmentConfig,
    pub output_dir: PathBuf,
    pub env_name: String,
}

#[derive(Debug, Clone, Copy)]
pub enum RlZooSchedule {
    Constant(f64),
    Linear(f64),
}

impl Serialize for RlZooSchedule {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Constant(value) => serializer.serialize_f64(*value),
            Self::Linear(value) => serializer.serialize_str(&format!("lin_{value}")),
        }
    }
}

impl RlZooSchedule {
    #[must_use]
    pub fn initial_value(self) -> f64 {
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

#[derive(Debug, Clone)]
pub enum RlZooNormalize {
    Enabled(bool),
    Options { norm_obs: bool, norm_reward: bool },
}

impl Serialize for RlZooNormalize {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Enabled(enabled) => serializer.serialize_bool(*enabled),
            Self::Options {
                norm_obs,
                norm_reward,
            } => serializer.serialize_str(&format!(
                "{{'norm_obs': {}, 'norm_reward': {}}}",
                python_bool(*norm_obs),
                python_bool(*norm_reward)
            )),
        }
    }
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

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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
}
