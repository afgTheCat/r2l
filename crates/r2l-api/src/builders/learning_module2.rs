use burn::{
    grad_clipping::{GradientClipping, GradientClippingConfig},
    optim::AdamWConfig,
    tensor::backend::AutodiffBackend,
};
use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_burn::{
    distributions::{
        PolicyKind,
        categorical_distribution::CategoricalDistribution as BurnCategoricalDistribution,
        diagonal_distribution::DiagGaussianDistribution as BurnDiagGaussianDistribution,
    },
    learning_module::{
        JointActorModel, JointPolicyValueModule, PolicyValueModuleKind as BurnPolicyValueModule,
        SplitPolicyValueModule,
    },
    sequential::Sequential,
};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::env::ActionSpaceType;

pub enum LearningModuleType {
    Joint {
        value_hidden_layers: Vec<usize>,
        max_grad_norm: Option<f32>,
    },
    Split {
        value_hidden_layers: Vec<usize>,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    },
}

struct LearningModuleBuilder2 {
    observation_size: usize,
    action_size: usize,
    action_space: ActionSpaceType,
    policy_hidden_layers: Vec<usize>,
    learning_module_type: LearningModuleType,
    params: ParamsAdamW,
}

impl LearningModuleBuilder2 {
    fn build_candle(
        &self,
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
        match &self.learning_module_type {
            LearningModuleType::Joint {
                value_hidden_layers: value_hidden_layer,
                max_grad_norm,
            } => CandlePolicyValueModule::build_joint2(
                policy,
                value_hidden_layer,
                policy_varmap,
                *max_grad_norm,
                self.params.clone(),
            ),
            LearningModuleType::Split {
                value_hidden_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
            } => CandlePolicyValueModule::build_split2(
                value_hidden_layers,
                policy,
                policy_varmap,
                *policy_max_grad_norm,
                *value_max_grad_norm,
                self.params.clone(),
            ),
        }
    }

    fn build_burn<B: AutodiffBackend>(self) -> anyhow::Result<BurnPolicyValueModule<B>> {
        let layers = &[&self.policy_hidden_layers[..], &[self.action_size]].concat();
        let policy_layers = &[&[self.observation_size][..], &layers[..]].concat();
        let policy = match self.action_space {
            ActionSpaceType::Discrete => {
                PolicyKind::Categorical(BurnCategoricalDistribution::<B>::build(policy_layers))
            }
            ActionSpaceType::Continuous => {
                PolicyKind::Diag(BurnDiagGaussianDistribution::build(policy_layers))
            }
        };
        let learning_module = match self.learning_module_type {
            LearningModuleType::Joint {
                value_hidden_layers: value_layers,
                max_grad_norm,
            } => {
                let value_layers =
                    &[&[self.observation_size][..], &value_layers[..], &[1]].concat();
                let value_net: Sequential<B> = Sequential::build(value_layers);
                let model = JointActorModel::new(policy, value_net);
                let mut optimizer = AdamWConfig::new()
                    .with_beta_1(self.params.beta1 as f32)
                    .with_beta_2(self.params.beta2 as f32)
                    .with_epsilon(self.params.eps as f32)
                    .with_weight_decay(self.params.weight_decay as f32);
                if let Some(max_grad_norm) = max_grad_norm {
                    optimizer = optimizer
                        .with_grad_clipping(Some(GradientClippingConfig::Norm(max_grad_norm)));
                }
                let model = JointPolicyValueModule::new(model, optimizer.init(), self.params.lr);
                BurnPolicyValueModule::Joint(model)
            }
            LearningModuleType::Split {
                value_hidden_layers: value_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
            } => {
                let value_layers =
                    &[&[self.observation_size][..], &value_layers[..], &[1]].concat();
                let value_net: Sequential<B> = Sequential::build(value_layers);
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
                let model = SplitPolicyValueModule::new(
                    policy,
                    value_net,
                    policy_optimizer.init(),
                    self.params.lr,
                    value_optimizer.init(),
                    self.params.lr,
                );
                BurnPolicyValueModule::Split(model)
            }
        };
        Ok(learning_module)
    }
}
