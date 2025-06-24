use crate::builders::policies::{PPODistributionKind, PolicyBuilder, PolicyType};
use candle_core::{Device, Result};
use r2l_agents::ppo::{PPO, hooks::PPOHooks};
use r2l_core::policies::PolicyKind;
use r2l_gym::GymEnv;

pub struct PPOBuilder {
    pub device: Device,
    pub distribution_kind: PPODistributionKind,
    pub policy_builder: PolicyBuilder,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for PPOBuilder {
    fn default() -> Self {
        PPOBuilder {
            device: Device::Cpu,
            distribution_kind: PPODistributionKind::DiagGaussianDistribution {
                hidden_layers: vec![64, 64],
            },
            policy_builder: PolicyBuilder {
                in_dim: 0,
                out_dim: 0,
                policy_type: PolicyType::Paralell {
                    value_layers: vec![64, 64],
                    max_grad_norm: None,
                },
            },
            clip_range: 0.2,
            lambda: 0.8,
            gamma: 0.98,
            sample_size: 64,
        }
    }
}

impl PPOBuilder {
    pub fn build(&self) -> Result<PPO<PolicyKind>> {
        let policy = self
            .policy_builder
            .build_policy(&self.distribution_kind, &self.device)?;
        Ok(PPO {
            policy,
            hooks: PPOHooks::empty(),
            clip_range: self.clip_range,
            device: self.device.clone(),
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size,
        })
    }

    pub fn set_gym_env_io(&mut self, env: &GymEnv) {
        let io_dim = env.io_sizes();
        self.policy_builder.set_io_dim(io_dim);
    }

    pub fn from_gym_env(env: &GymEnv) -> Self {
        let (in_dim, out_dim) = env.io_sizes();
        PPOBuilder {
            policy_builder: PolicyBuilder {
                in_dim,
                out_dim,
                policy_type: PolicyType::Paralell {
                    value_layers: vec![64, 64],
                    max_grad_norm: None,
                },
            },
            ..Default::default()
        }
    }
}
