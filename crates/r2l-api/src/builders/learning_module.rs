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

pub enum LearningModuleType {
    Joint {
        max_grad_norm: Option<f32>,
    },
    Split {
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    },
}

pub struct LearningModuleBuilder {
    pub policy_hidden_layers: Vec<usize>,
    pub value_hidden_layers: Vec<usize>,
    pub learning_module_type: LearningModuleType,
    pub params: ParamsAdamW,
}

impl LearningModuleBuilder {
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
            LearningModuleType::Joint { max_grad_norm } => CandlePolicyValueModule::build_joint(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                max_grad_norm,
                self.params.clone(),
            ),
            LearningModuleType::Split {
                policy_max_grad_norm,
                value_max_grad_norm,
            } => CandlePolicyValueModule::build_split(
                &self.value_hidden_layers,
                policy,
                policy_varmap,
                policy_max_grad_norm,
                value_max_grad_norm,
                self.params.clone(),
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
            LearningModuleType::Joint { max_grad_norm } => {
                let value_layers =
                    &[&[observation_size][..], &self.value_hidden_layers[..], &[1]].concat();
                let mut optimizer_config = AdamWConfig::new()
                    .with_beta_1(self.params.beta1 as f32)
                    .with_beta_2(self.params.beta2 as f32)
                    .with_epsilon(self.params.eps as f32)
                    .with_weight_decay(self.params.weight_decay as f32);
                if let Some(max_grad_norm) = max_grad_norm {
                    optimizer_config = optimizer_config
                        .with_grad_clipping(Some(GradientClippingConfig::Norm(max_grad_norm)));
                }
                BurnPolicyValueModule::joint(policy, value_layers, optimizer_config, self.params.lr)
            }
            LearningModuleType::Split {
                policy_max_grad_norm,
                value_max_grad_norm,
            } => {
                let value_layers =
                    &[&[observation_size][..], &self.value_hidden_layers[..], &[1]].concat();
                let mut policy_optimizer = AdamWConfig::new()
                    .with_beta_1(self.params.beta1 as f32)
                    .with_beta_2(self.params.beta2 as f32)
                    .with_epsilon(self.params.eps as f32)
                    .with_weight_decay(self.params.weight_decay as f32);
                if let Some(policy_max_grad_norm) = policy_max_grad_norm {
                    policy_optimizer = policy_optimizer.with_grad_clipping(Some(
                        GradientClippingConfig::Norm(policy_max_grad_norm),
                    ));
                }
                let mut value_optimizer = AdamWConfig::new()
                    .with_beta_1(self.params.beta1 as f32)
                    .with_beta_2(self.params.beta2 as f32)
                    .with_epsilon(self.params.eps as f32)
                    .with_weight_decay(self.params.weight_decay as f32);
                if let Some(value_max_grad_norm) = value_max_grad_norm {
                    value_optimizer = value_optimizer.with_grad_clipping(Some(
                        GradientClippingConfig::Norm(value_max_grad_norm),
                    ));
                }
                BurnPolicyValueModule::split(
                    policy,
                    value_layers,
                    policy_optimizer,
                    self.params.lr,
                    value_optimizer,
                    self.params.lr,
                )
            }
        };
        Ok(learning_module)
    }
}
