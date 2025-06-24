use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use r2l_core::{
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

pub enum PolicyType {
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

pub enum PPODistributionKind {
    CategoricalDistribution {
        action_size: usize,
        hidden_layers: Vec<usize>,
    },
    DiagGaussianDistribution {
        hidden_layers: Vec<usize>,
    },
}

impl PPODistributionKind {
    pub fn build(
        &self,
        vb: &VarBuilder,
        input_dim: usize,
        out_dim: usize,
        device: &Device,
    ) -> Result<DistributionKind> {
        match &self {
            Self::DiagGaussianDistribution { hidden_layers } => {
                let log_std = vb.get(out_dim, "log_std")?;
                let layers = &[&hidden_layers[..], &[out_dim]].concat();
                let distr =
                    DiagGaussianDistribution::build(input_dim, layers, &vb, log_std, "policy")?;
                Ok(DistributionKind::DiagGaussian(distr))
            }
            Self::CategoricalDistribution {
                action_size,
                hidden_layers,
            } => {
                let layers = &[&hidden_layers[..], &[out_dim]].concat();
                let distr = CategoricalDistribution::build(
                    input_dim,
                    *action_size,
                    layers,
                    &vb,
                    device.clone(),
                    "policy",
                )?;
                Ok(DistributionKind::Categorical(distr))
            }
        }
    }
}

pub struct PolicyBuilder {
    pub in_dim: usize,
    pub out_dim: usize,
    pub policy_type: PolicyType,
}

impl PolicyBuilder {
    pub fn build_policy(
        &self,
        distribuition_kind: &PPODistributionKind,
        device: &Device,
    ) -> Result<PolicyKind> {
        assert!(self.in_dim > 0, "Input dim has to be larger than 0");
        assert!(self.out_dim > 0, "Output dim has to be larger than 0");
        let optimizer_params = ParamsAdamW {
            lr: 0.001,
            weight_decay: 0.01,
            ..Default::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let distribution = distribuition_kind.build(&vb, self.in_dim, self.out_dim, &device)?;
        match &self.policy_type {
            PolicyType::Paralell {
                value_layers,
                max_grad_norm,
            } => {
                let value_layers = &[&value_layers[..], &[1]].concat();
                let (value_net, _) = build_sequential(self.in_dim, value_layers, &vb, "value")?;
                let optimizer = AdamW::new(varmap.all_vars(), optimizer_params.clone())?;
                let policy = ParalellActorCritic::new(
                    distribution,
                    value_net,
                    optimizer,
                    *max_grad_norm,
                    varmap,
                );
                Ok(PolicyKind::Paralell(policy))
            }
            PolicyType::Decoupled {
                value_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
            } => {
                let critic_varmap = VarMap::new();
                let critic_vb = VarBuilder::from_varmap(&critic_varmap, DType::F32, &device);
                let value_layers = &[&value_layers[..], &[1]].concat();
                let (value_net, _) =
                    build_sequential(self.in_dim, value_layers, &critic_vb, "value")?;
                let policy_optimizer = AdamW::new(varmap.all_vars(), optimizer_params.clone())?;
                let value_optimizer = AdamW::new(critic_varmap.all_vars(), optimizer_params)?;
                let policy = DecoupledActorCritic {
                    distribution,
                    value_net,
                    policy_optimizer,
                    value_optimizer,
                    policy_max_grad_norm: *policy_max_grad_norm,
                    value_max_grad_norm: *value_max_grad_norm,
                    policy_varmap: varmap,
                    value_varmap: critic_varmap,
                };
                Ok(PolicyKind::Decoupled(policy))
            }
        }
    }

    pub fn set_io_dim(&mut self, (in_dim, out_dim): (usize, usize)) {
        self.in_dim = in_dim;
        self.out_dim = out_dim;
    }
}
