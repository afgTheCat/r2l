use burn::{
    grad_clipping::GradientClippingConfig, optim::AdamWConfig, tensor::backend::AutodiffBackend,
};
use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_burn::{
    distributions::PolicyKind, learning_module::PolicyValueModuleKind as BurnPolicyValueModule,
};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::env::ActionSpaceType;

pub enum OnPolicyLearningModuleType {
    Joint {
        max_grad_norm: Option<f32>,
        params: ParamsAdamW,
    },
    Split {
        policy_max_grad_norm: Option<f32>,
        policy_params: ParamsAdamW,
        value_max_grad_norm: Option<f32>,
        value_params: ParamsAdamW,
    },
}

impl OnPolicyLearningModuleType {
    fn map_params<F>(self, mut f: F) -> Self
    where
        F: FnMut(&mut ParamsAdamW),
    {
        match self {
            Self::Joint {
                max_grad_norm,
                mut params,
            } => {
                f(&mut params);
                Self::Joint {
                    params,
                    max_grad_norm,
                }
            }
            Self::Split {
                policy_max_grad_norm,
                mut policy_params,
                value_max_grad_norm,
                mut value_params,
            } => {
                f(&mut policy_params);
                f(&mut value_params);
                Self::Split {
                    policy_max_grad_norm,
                    policy_params,
                    value_max_grad_norm,
                    value_params,
                }
            }
        }
    }

    pub fn with_lr(self, lr: f64) -> Self {
        self.map_params(|params| params.lr = lr)
    }

    pub fn with_beta1(self, beta1: f64) -> Self {
        self.map_params(|params| params.beta1 = beta1)
    }

    pub fn with_beta2(self, beta2: f64) -> Self {
        self.map_params(|params| params.beta2 = beta2)
    }

    pub fn with_epsilon(self, epsilon: f64) -> Self {
        self.map_params(|params| params.eps = epsilon)
    }

    pub fn with_weight_decay(self, weight_decay: f64) -> Self {
        self.map_params(|params| params.weight_decay = weight_decay)
    }
}

pub(crate) struct OnPolicyLearningModuleBuilder {
    pub(crate) policy_hidden_layers: Vec<usize>,
    pub(crate) value_hidden_layers: Vec<usize>,
    pub(crate) learning_module_type: OnPolicyLearningModuleType,
}

impl OnPolicyLearningModuleBuilder {
    pub fn build_candle(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: &Device,
    ) -> anyhow::Result<CandlePolicyValueModule> {
        let policy_varmap = VarMap::new();
        let policy_vb = VarBuilder::from_varmap(&policy_varmap, DType::F32, device);
        let policy = CandlePolicyKind::build(
            action_space,
            &policy_vb,
            &self.policy_hidden_layers,
            action_size,
            observation_size,
        )?;
        match self.learning_module_type {
            OnPolicyLearningModuleType::Joint {
                max_grad_norm,
                params,
            } => CandlePolicyValueModule::build_joint(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                max_grad_norm,
                params,
            ),
            OnPolicyLearningModuleType::Split {
                policy_max_grad_norm,
                policy_params,
                value_max_grad_norm,
                value_params,
            } => CandlePolicyValueModule::build_split(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                policy_max_grad_norm,
                value_max_grad_norm,
                policy_params,
                value_params,
            ),
        }
    }

    pub fn build_burn<B: AutodiffBackend>(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<BurnPolicyValueModule<B>> {
        let policy_layers = &[
            &[observation_size][..],
            &self.policy_hidden_layers[..],
            &[action_size],
        ]
        .concat();
        let policy = PolicyKind::build(action_space, policy_layers);
        let learning_module = match self.learning_module_type {
            OnPolicyLearningModuleType::Joint {
                max_grad_norm,
                params,
            } => {
                let value_layers =
                    &[&[observation_size][..], &self.value_hidden_layers[..], &[1]].concat();
                let mut optimizer_config = AdamWConfig::new()
                    .with_beta_1(params.beta1 as f32)
                    .with_beta_2(params.beta2 as f32)
                    .with_epsilon(params.eps as f32)
                    .with_weight_decay(params.weight_decay as f32);
                if let Some(max_grad_norm) = max_grad_norm {
                    optimizer_config = optimizer_config
                        .with_grad_clipping(Some(GradientClippingConfig::Norm(max_grad_norm)));
                }
                BurnPolicyValueModule::joint(policy, value_layers, optimizer_config, params.lr)
            }
            OnPolicyLearningModuleType::Split {
                policy_max_grad_norm,
                policy_params,
                value_max_grad_norm,
                value_params,
            } => {
                let value_layers =
                    &[&[observation_size][..], &self.value_hidden_layers[..], &[1]].concat();
                let mut policy_optimizer = AdamWConfig::new()
                    .with_beta_1(policy_params.beta1 as f32)
                    .with_beta_2(policy_params.beta2 as f32)
                    .with_epsilon(policy_params.eps as f32)
                    .with_weight_decay(policy_params.weight_decay as f32);
                if let Some(policy_max_grad_norm) = policy_max_grad_norm {
                    policy_optimizer = policy_optimizer.with_grad_clipping(Some(
                        GradientClippingConfig::Norm(policy_max_grad_norm),
                    ));
                }
                let mut value_optimizer = AdamWConfig::new()
                    .with_beta_1(value_params.beta1 as f32)
                    .with_beta_2(value_params.beta2 as f32)
                    .with_epsilon(value_params.eps as f32)
                    .with_weight_decay(value_params.weight_decay as f32);
                if let Some(value_max_grad_norm) = value_max_grad_norm {
                    value_optimizer = value_optimizer.with_grad_clipping(Some(
                        GradientClippingConfig::Norm(value_max_grad_norm),
                    ));
                }
                BurnPolicyValueModule::split(
                    policy,
                    value_layers,
                    policy_optimizer,
                    policy_params.lr,
                    value_optimizer,
                    value_params.lr,
                )
            }
        };
        Ok(learning_module)
    }
}
