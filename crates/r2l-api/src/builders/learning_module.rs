use burn::{optim::AdamWConfig, tensor::backend::AutodiffBackend};
use candle_nn::ParamsAdamW;
use r2l_burn::{
    learning_module::{
        BurnPolicy, JointActorModel, JointPolicyValueModule, PolicyValueModule,
        SplitPolicyValueModule,
    },
    sequential::Sequential,
};

pub enum LearningModuleType {
    Joint {
        value_layers: Vec<usize>,
        max_grad_norm: Option<f32>,
    },
    Split {
        value_layers: Vec<usize>,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    },
}

// TODO: we probably need to rethink this. We need to add the optimizer params
pub struct LearningModuleBuilder {
    pub learning_module_type: LearningModuleType,
    pub params: ParamsAdamW,
}

impl LearningModuleBuilder {
    pub fn build_burn<B: AutodiffBackend, D: BurnPolicy<B>>(
        &self,
        observation_size: usize,
        policy: D,
    ) -> PolicyValueModule<B, D> {
        match &self.learning_module_type {
            LearningModuleType::Joint {
                value_layers,
                // TODO: should we consider this
                max_grad_norm: _max_grad_norm,
            } => {
                let value_layers = &[&[observation_size][..], &value_layers[..], &[1]].concat();
                let value_net: Sequential<B> = Sequential::build(value_layers);
                let model = JointActorModel::new(policy, value_net);
                PolicyValueModule::Joint(JointPolicyValueModule::new(
                    model,
                    AdamWConfig::new()
                        .with_beta_1(self.params.beta1 as f32)
                        .with_beta_2(self.params.beta2 as f32)
                        .with_epsilon(self.params.eps as f32)
                        .with_weight_decay(self.params.weight_decay as f32)
                        .init(),
                    self.params.lr,
                ))
            }
            LearningModuleType::Split {
                value_layers,
                // TODO: should we consider this
                policy_max_grad_norm: _policy_max_grad_norm,
                value_max_grad_norm: _value_max_grad_norm,
            } => {
                let value_layers = &[&[observation_size][..], &value_layers[..], &[1]].concat();
                let value_net: Sequential<B> = Sequential::build(value_layers);
                PolicyValueModule::Split(SplitPolicyValueModule::new(
                    policy,
                    value_net,
                    AdamWConfig::new()
                        .with_beta_1(self.params.beta1 as f32)
                        .with_beta_2(self.params.beta2 as f32)
                        .with_epsilon(self.params.eps as f32)
                        .with_weight_decay(self.params.weight_decay as f32)
                        .init(),
                    self.params.lr,
                    AdamWConfig::new()
                        .with_beta_1(self.params.beta1 as f32)
                        .with_beta_2(self.params.beta2 as f32)
                        .with_epsilon(self.params.eps as f32)
                        .with_weight_decay(self.params.weight_decay as f32)
                        .init(),
                    self.params.lr,
                ))
            }
        }
    }
}
