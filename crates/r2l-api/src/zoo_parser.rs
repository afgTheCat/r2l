use std::{collections::BTreeMap, fs, path::PathBuf};

use serde::{Deserialize, Serialize};
use yaml_serde::Value;

#[derive(Debug, Serialize, Deserialize)]
pub struct RlZooDefault {
    n_timesteps: f32,
    policy: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RlZooEnvironmentConfig {
    n_envs: Option<usize>,
    n_timesteps: Option<f32>,
    policy: Option<String>,
    n_steps: Option<usize>,
    gae_lambda: Option<f32>,
    gamma: Option<f32>,
    n_epochs: Option<usize>,
    ent_coef: Option<f32>,
    learning_rate: Option<Value>,
    clip_range: Option<Value>,
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

    // fn to_algorithm(self) -> Option<> {}
}

pub fn parse_rl_zoo_config(path: PathBuf) {
    let content = fs::read_to_string(path).unwrap();
    let mut parsed_content: BTreeMap<String, Value> = yaml_serde::from_str(&content).unwrap();
    let rl_zoo_default: RlZooDefault = parsed_content
        .remove("default")
        .map(|val| yaml_serde::from_value(val).unwrap())
        .unwrap();
    // TODO: should we pass this?
    parsed_content.remove("atari");
    let zoo_configs: BTreeMap<String, RlZooEnvironmentConfig> = parsed_content
        .into_iter()
        .map(|(env_name, val)| {
            let rl_zoo_config = yaml_serde::from_value::<RlZooEnvironmentConfig>(val).unwrap();
            let rl_zoo_config = rl_zoo_config.merge_with_default(&rl_zoo_default);
            (env_name, rl_zoo_config)
        })
        .collect();
    println!("{zoo_configs:#?}");
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::zoo_parser::parse_rl_zoo_config;

    #[test]
    fn zoo_parser_test() {
        let path = PathBuf::from("/home/g/git/r2l/assets/ppo.yaml");
        parse_rl_zoo_config(path);
    }
}
