use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use r2l_core::{
    distributions::{
        DistributionKind, categorical_distribution::CategoricalDistribution,
        diagonal_distribution::DiagGaussianDistribution,
    },
    env::{EnvironmentDescription, Space},
    policies::{
        OptimizerWithMaxGrad, PolicyKind, decoupled_actor_critic::DecoupledActorCritic,
        paralell_actor_critic::ParalellActorCritic,
    },
    thread_safe_sequential::build_sequential,
};
use r2l_gym::GymEnv;

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
    Dynamic { hidden_layers: Vec<usize> },
    CategoricalDistribution { hidden_layers: Vec<usize> },
    DiagGaussianDistribution { hidden_layers: Vec<usize> },
}

impl PPODistributionKind {
    fn build_diag_gaussian(
        vb: &VarBuilder,
        env_description: &EnvironmentDescription,
        hidden_layers: &[usize],
    ) -> Result<DistributionKind> {
        let action_size = env_description.action_size();
        let observation_size = env_description.observation_size();
        let layers = &[&hidden_layers[..], &[action_size]].concat();
        let log_std = vb.get(action_size, "log_std")?;
        let distr =
            DiagGaussianDistribution::build(observation_size, layers, &vb, log_std, "policy")?;
        Ok(DistributionKind::DiagGaussian(distr))
    }

    fn build_categorical(
        vb: &VarBuilder,
        env_description: &EnvironmentDescription,
        device: &Device,
        hidden_layers: &[usize],
    ) -> Result<DistributionKind> {
        let action_size = env_description.action_size();
        let obseravation_size = env_description.observation_size();
        let layers = &[&hidden_layers[..], &[action_size]].concat();
        let distr = CategoricalDistribution::build(
            obseravation_size,
            action_size,
            layers,
            &vb,
            device.clone(),
            "policy",
        )?;
        Ok(DistributionKind::Categorical(distr))
    }

    pub fn build(
        &self,
        vb: &VarBuilder,
        device: &Device,
        env_description: &EnvironmentDescription,
    ) -> Result<DistributionKind> {
        match &self {
            Self::DiagGaussianDistribution { hidden_layers } => {
                Self::build_diag_gaussian(vb, env_description, hidden_layers)
            }
            Self::CategoricalDistribution { hidden_layers } => {
                Self::build_categorical(vb, env_description, device, hidden_layers)
            }
            Self::Dynamic { hidden_layers } => match env_description.action_space {
                Space::Discrete(..) => {
                    Self::build_categorical(vb, env_description, device, &hidden_layers)
                }
                Space::Continous { .. } => {
                    Self::build_diag_gaussian(vb, env_description, hidden_layers)
                }
            },
        }
    }

    pub fn from_env(env: &GymEnv) -> Self {
        match env.action_space() {
            Space::Discrete(..) => PPODistributionKind::CategoricalDistribution {
                hidden_layers: vec![64, 64],
            },
            Space::Continous { .. } => PPODistributionKind::DiagGaussianDistribution {
                hidden_layers: vec![64, 64],
            },
        }
    }
}

pub struct PolicyBuilder {
    pub policy_type: PolicyType,
}

impl PolicyBuilder {
    pub fn build_policy(
        &self,
        distribution_kind: &PPODistributionKind,
        env_description: &EnvironmentDescription,
        device: &Device,
    ) -> Result<PolicyKind> {
        let input_size = env_description.observation_size();
        let optimizer_params = ParamsAdamW {
            lr: 3e-4,
            weight_decay: 0.01,
            ..Default::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let distribution = distribution_kind.build(&vb, &device, env_description)?;
        match &self.policy_type {
            PolicyType::Paralell {
                value_layers,
                max_grad_norm,
            } => {
                let value_layers = &[&value_layers[..], &[1]].concat();
                let value_net = build_sequential(input_size, value_layers, &vb, "value")?;
                let optimizer = AdamW::new(varmap.all_vars(), optimizer_params.clone())?;
                let optimizer_with_grad =
                    OptimizerWithMaxGrad::new(optimizer, *max_grad_norm, varmap);
                let policy = ParalellActorCritic::new(distribution, value_net, optimizer_with_grad);
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
                let value_net = build_sequential(input_size, value_layers, &critic_vb, "value")?;
                let policy_optimizer = AdamW::new(varmap.all_vars(), optimizer_params.clone())?;
                let value_optimizer = AdamW::new(critic_varmap.all_vars(), optimizer_params)?;
                let policy_optimizer_with_grad =
                    OptimizerWithMaxGrad::new(policy_optimizer, *policy_max_grad_norm, varmap);
                let value_optimizer_with_grad =
                    OptimizerWithMaxGrad::new(value_optimizer, *value_max_grad_norm, critic_varmap);
                let policy = DecoupledActorCritic {
                    distribution,
                    value_net,
                    policy_optimizer_with_grad,
                    value_optimizer_with_grad,
                };
                Ok(PolicyKind::Decoupled(policy))
            }
        }
    }
}
