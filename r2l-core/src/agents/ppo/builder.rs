use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::{
    distributions::{
        DistributionKind, categorical_distribution::CategoricalDistribution,
        diagonal_distribution::DiagGaussianDistribution,
    },
    policies::{
        PolicyKind, decoupled_actor_critic::DecoupledActorCritic,
        paralell_actor_critic::ParalellActorCritic,
    },
    utils::build_sequential::build_sequential,
};

use super::{PPO, hooks::PPOHooks};

pub enum PPODistributionKind {
    CategoricalDistribution {
        action_size: usize,
        hidden_layers: Vec<usize>,
    },
    DiagGaussianDistribution {
        hidden_layers: Vec<usize>,
    },
}

pub enum PPOPolicyType {
    Paralell {
        value_layers: Vec<usize>,
        max_grad_norm: Option<f32>,
    },
    Decoupled {
        value_layers: Vec<usize>,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    },
}

// TODO: we might not need a builder here after al
pub struct PPOBuilder {
    pub device: Device,
    pub input_dim: usize,
    pub out_dim: usize,
    pub policy_type: PPOPolicyType,
    pub distribution_type: PPODistributionKind,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for PPOBuilder {
    fn default() -> Self {
        PPOBuilder {
            device: Device::Cpu,
            input_dim: 0,
            out_dim: 0,
            policy_type: PPOPolicyType::Paralell {
                value_layers: vec![64, 64],
                max_grad_norm: None,
            },
            distribution_type: PPODistributionKind::DiagGaussianDistribution {
                hidden_layers: vec![64, 64],
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
        let optimizer_params = ParamsAdamW {
            lr: 0.001,
            weight_decay: 0.01,
            ..Default::default()
        };
        let policy = match &self.policy_type {
            PPOPolicyType::Paralell {
                value_layers,
                max_grad_norm,
            } => {
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
                let distribution = match &self.distribution_type {
                    PPODistributionKind::DiagGaussianDistribution { hidden_layers } => {
                        let log_std = vb.get(self.out_dim, "log_std")?;
                        let layers = &[&hidden_layers[..], &[self.out_dim]].concat();
                        let distr = DiagGaussianDistribution::build(
                            self.input_dim,
                            layers,
                            &vb,
                            log_std,
                            "policy",
                        )?;
                        DistributionKind::DiagGaussian(distr)
                    }
                    PPODistributionKind::CategoricalDistribution {
                        action_size,
                        hidden_layers,
                    } => {
                        let layers = &[&hidden_layers[..], &[self.out_dim]].concat();
                        let distr = CategoricalDistribution::build(
                            self.input_dim,
                            *action_size,
                            layers,
                            &vb,
                            self.device.clone(),
                            "policy",
                        )?;
                        DistributionKind::Categorical(distr)
                    }
                };
                let value_layers = &[&value_layers[..], &[1]].concat();
                let (value_net, _) = build_sequential(self.input_dim, value_layers, &vb, "value")?;
                let optimizer = AdamW::new(varmap.all_vars(), optimizer_params.clone())?;
                let policy = ParalellActorCritic::new(
                    distribution,
                    value_net,
                    optimizer,
                    *max_grad_norm,
                    varmap,
                );
                PolicyKind::Paralell(policy)
            }
            PPOPolicyType::Decoupled {
                value_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
            } => {
                let policy_varmap = VarMap::new();
                let critic_varmap = VarMap::new();
                let policy_vb = VarBuilder::from_varmap(&policy_varmap, DType::F32, &self.device);
                let critic_vb = VarBuilder::from_varmap(&critic_varmap, DType::F32, &self.device);
                let distribution = match &self.distribution_type {
                    PPODistributionKind::DiagGaussianDistribution { hidden_layers } => {
                        let log_std = policy_vb.get(self.out_dim, "log_std")?;
                        let layers = &[&hidden_layers[..], &[self.out_dim]].concat();
                        let distr = DiagGaussianDistribution::build(
                            self.input_dim,
                            layers,
                            &policy_vb,
                            log_std,
                            "policy",
                        )?;
                        DistributionKind::DiagGaussian(distr)
                    }
                    PPODistributionKind::CategoricalDistribution {
                        action_size,
                        hidden_layers,
                    } => {
                        let layers = &[&hidden_layers[..], &[self.out_dim]].concat();
                        let distr = CategoricalDistribution::build(
                            self.input_dim,
                            *action_size,
                            layers,
                            &policy_vb,
                            self.device.clone(),
                            "policy",
                        )?;
                        DistributionKind::Categorical(distr)
                    }
                };
                let value_layers = &[&value_layers[..], &[1]].concat();
                let (value_net, _) =
                    build_sequential(self.input_dim, value_layers, &critic_vb, "value")?;
                let policy_optimizer =
                    AdamW::new(policy_varmap.all_vars(), optimizer_params.clone())?;
                let value_optimizer = AdamW::new(critic_varmap.all_vars(), optimizer_params)?;
                let policy = DecoupledActorCritic {
                    distribution,
                    value_net,
                    policy_optimizer,
                    value_optimizer,
                    policy_max_grad_norm: *policy_max_grad_norm,
                    value_max_grad_norm: *value_max_grad_norm,
                    policy_varmap,
                    value_varmap: critic_varmap,
                };
                PolicyKind::Decoupled(policy)
            }
        };
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
}
