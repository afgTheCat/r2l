//! Serializable task definitions shared by the benchmark scheduler and worker.

use std::num::NonZeroUsize;

use r2l::{BackendConfig, OnPolicyConfig, PPOConfig};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct R2lTask {
    pub backend: BackendConfig,
    pub ppo: PPOConfig,
    pub config: OnPolicyConfig,
    pub env_name: String,
    pub n_envs: NonZeroUsize,
}

#[derive(Serialize, Deserialize)]
pub enum BenchmarkTask {
    R2l(Box<R2lTask>),
    Sb3 { env_name: String },
}
