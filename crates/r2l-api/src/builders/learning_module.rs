use burn::{
    grad_clipping::GradientClippingConfig, optim::AdamWConfig, tensor::backend::AutodiffBackend,
};
use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_burn::learning_module::ActionSpacePolicyValueModule as BurnPolicyValueModule;
use r2l_candle::learning_module::PolicyValueModule as CandlePolicyValueModule;
use r2l_core::{env::Space, tensor::R2lTensor};
use serde::{Deserialize, Serialize};

use crate::builders::policy::PolicyBuilder;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamWParams {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl AdamWParams {
    fn into_candle_params(self) -> ParamsAdamW {
        ParamsAdamW {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
}

/// Optimizer layout for on-policy policy/value learning modules.
///
/// This controls whether policy and value learning share a single optimizer
/// configuration or use separate optimizer configurations.
#[derive(Debug, Serialize, Deserialize)]
pub enum OnPolicyOptimizerLayout {
    /// Use one joint optimizer configuration for both policy and value updates.
    Joint {
        /// Optional global gradient-norm clipping threshold.
        max_grad_norm: Option<f32>,
        /// Shared AdamW optimizer parameters.
        params: AdamWParams,
    },
    /// Use separate optimizer configurations for policy and value updates.
    Split {
        /// Optional policy gradient-norm clipping threshold.
        policy_max_grad_norm: Option<f32>,
        /// Policy AdamW optimizer parameters.
        policy_params: AdamWParams,
        /// Optional value-function gradient-norm clipping threshold.
        value_max_grad_norm: Option<f32>,
        /// Value-function AdamW optimizer parameters.
        value_params: AdamWParams,
    },
}

impl OnPolicyOptimizerLayout {
    /// Returns a copy with the learning rate updated everywhere it applies.
    fn map_params<F>(self, mut f: F) -> Self
    where
        F: FnMut(&mut AdamWParams),
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

    /// Sets the AdamW learning rate on all contained optimizer configs.
    pub fn with_lr(self, lr: f64) -> Self {
        self.map_params(|params| params.lr = lr)
    }

    /// Sets the AdamW `beta1` parameter on all contained optimizer configs.
    pub fn with_beta1(self, beta1: f64) -> Self {
        self.map_params(|params| params.beta1 = beta1)
    }

    /// Sets the AdamW `beta2` parameter on all contained optimizer configs.
    pub fn with_beta2(self, beta2: f64) -> Self {
        self.map_params(|params| params.beta2 = beta2)
    }

    /// Sets the AdamW epsilon parameter on all contained optimizer configs.
    pub fn with_epsilon(self, epsilon: f64) -> Self {
        self.map_params(|params| params.eps = epsilon)
    }

    /// Sets the AdamW weight decay on all contained optimizer configs.
    pub fn with_weight_decay(self, weight_decay: f64) -> Self {
        self.map_params(|params| params.weight_decay = weight_decay)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OnPolicyLearningModuleBuilder {
    pub(crate) policy_builder: PolicyBuilder,
    pub(crate) value_hidden_layers: Vec<usize>,
    pub(crate) optimizer_layout: OnPolicyOptimizerLayout,
}

impl OnPolicyLearningModuleBuilder {
    pub fn build_candle<T: R2lTensor>(
        self,
        observation_size: usize,
        action_space: Space<T>,
        device: &Device,
    ) -> anyhow::Result<CandlePolicyValueModule> {
        let (policy, policy_varmap) =
            self.policy_builder
                .build_candle_with_varmap(observation_size, action_space, device)?;
        let activation_function = self.policy_builder.activation_function;
        match self.optimizer_layout {
            OnPolicyOptimizerLayout::Joint {
                max_grad_norm,
                params,
            } => CandlePolicyValueModule::build_joint(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                max_grad_norm,
                params.into_candle_params(),
                activation_function,
            ),
            OnPolicyOptimizerLayout::Split {
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
                policy_params.into_candle_params(),
                value_params.into_candle_params(),
                activation_function,
            ),
        }
    }

    pub fn build_burn<B: AutodiffBackend, T: R2lTensor>(
        self,
        observation_size: usize,
        action_space: Space<T>,
    ) -> anyhow::Result<BurnPolicyValueModule<B>> {
        let policy = self
            .policy_builder
            .build_burn::<B, _>(observation_size, action_space);
        let activation_function = self.policy_builder.activation_function;
        let learning_module = match self.optimizer_layout {
            OnPolicyOptimizerLayout::Joint {
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
                BurnPolicyValueModule::joint(
                    policy,
                    value_layers,
                    activation_function,
                    optimizer_config,
                    params.lr,
                )
            }
            OnPolicyOptimizerLayout::Split {
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
                    activation_function,
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
