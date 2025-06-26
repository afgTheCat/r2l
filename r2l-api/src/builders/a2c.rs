use crate::builders::policies::{PPODistributionKind, PolicyBuilder, PolicyType};
use candle_core::{Device, Result};
use r2l_agents::a2c::{A2C, hooks::A2CHooks};
use r2l_core::{env::EnvironmentDescription, policies::PolicyKind};

pub struct A2CBuilder {
    pub distribution_kind: PPODistributionKind,
    pub policy_builder: PolicyBuilder,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for A2CBuilder {
    fn default() -> Self {
        A2CBuilder {
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

impl A2CBuilder {
    pub fn build(
        &self,
        device: &Device,
        env_description: &EnvironmentDescription,
    ) -> Result<A2C<PolicyKind>> {
        let policy =
            self.policy_builder
                .build_policy(&self.distribution_kind, env_description, &device)?;
        Ok(A2C {
            policy,
            hooks: A2CHooks::empty(),
            device: device.clone(),
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size,
        })
    }
}
