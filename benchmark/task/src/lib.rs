//! Serializable task definitions shared by the benchmark scheduler and worker.

use std::num::NonZeroUsize;

use r2l::{
    AlgoBackendConfig, BackendConfig, EnvBuilderType, GymEnvBuilder, OnPolicyBuilder,
    OnPolicyConfig, PPOConfig, error::Result,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct R2lTask {
    pub backend: BackendConfig,
    pub ppo: PPOConfig,
    pub config: OnPolicyConfig,
    pub env_name: String,
    pub n_envs: NonZeroUsize,
}

impl R2lTask {
    pub fn execute(self) -> Result<()> {
        let env_builder =
            EnvBuilderType::homogeneous(GymEnvBuilder::new(&self.env_name), self.n_envs.get())?;
        match self.backend {
            BackendConfig::Burn(backend) => {
                let algo_backend = AlgoBackendConfig {
                    algo: self.ppo,
                    backend,
                };
                OnPolicyBuilder::from_config(self.config, algo_backend, env_builder)
                    .build()?
                    .train()
            }
            BackendConfig::Candle(backend) => {
                let algo_backend = AlgoBackendConfig {
                    algo: self.ppo,
                    backend,
                };
                OnPolicyBuilder::from_config(self.config, algo_backend, env_builder)
                    .build()?
                    .train()
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum BenchmarkTask {
    R2l(Box<R2lTask>),
    Sb3 { env_name: String },
}
