use crate::builders::policies::{PPODistributionKind, PolicyBuilder, PolicyType};
use candle_core::{Device, Result};
use r2l_agents::ppo::{PPO, hooks::PPOHooks};
use r2l_core::{env::EnvironmentDescription, policies::PolicyKind};

pub struct PPOBuilder {
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
            distribution_kind: PPODistributionKind::Dynamic {
                hidden_layers: vec![64, 64],
            },
            policy_builder: PolicyBuilder {
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
    pub fn build(
        &self,
        device: &Device,
        env_description: &EnvironmentDescription,
    ) -> Result<PPO<PolicyKind>> {
        let policy =
            self.policy_builder
                .build_policy(&self.distribution_kind, env_description, &device)?;
        Ok(PPO {
            policy,
            hooks: PPOHooks::empty(),
            clip_range: self.clip_range,
            device: device.clone(),
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size,
        })
    }
}
