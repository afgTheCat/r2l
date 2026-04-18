use burn::{optim::AdamWConfig, prelude::Backend, tensor::backend::AutodiffBackend};
use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_burn::{
    distributions::{
        PolicyKind,
        categorical_distribution::CategoricalDistribution as BurnCategoricalDistribution,
        diagonal_distribution::DiagGaussianDistribution as BurnDiagGaussianDistribution,
    },
    learning_module::{
        JointActorModel, JointPolicyValueModule, PolicyValueModule,
        PolicyValueModuleKind as BurnPolicyValueModule, SplitPolicyValueModule,
    },
    sequential::Sequential,
};
use r2l_candle::learning_module::PolicyValueModule as CandlePolicyValueModule;
use r2l_core::env::ActionSpaceType;

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

struct LearningModuleBuilder {
    observation_size: usize,
    action_size: usize,
    action_space: ActionSpaceType,
    policy_hidden_layers: Vec<usize>,
    learning_module_type: LearningModuleType,
    params: ParamsAdamW,
}

impl LearningModuleBuilder {
    fn build_candle(self, device: &Device) -> anyhow::Result<CandlePolicyValueModule> {
        match self.learning_module_type {
            LearningModuleType::Joint {
                value_layers,
                max_grad_norm,
            } => CandlePolicyValueModule::build_joint(
                device,
                self.action_size,
                self.observation_size,
                &self.policy_hidden_layers,
                self.action_space,
                &value_layers,
                max_grad_norm,
                self.params,
            ),
            LearningModuleType::Split {
                value_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
            } => CandlePolicyValueModule::build_split(
                device,
                self.action_size,
                self.observation_size,
                &self.policy_hidden_layers,
                self.action_space,
                &value_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
                self.params,
            ),
        }
    }

    fn build_burn<B: AutodiffBackend>(&self) -> anyhow::Result<BurnPolicyValueModule<B>> {
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
                value_layers,
                max_grad_norm,
            } => {
                let value_layers =
                    &[&[self.observation_size][..], &value_layers[..], &[1]].concat();
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
                policy_max_grad_norm,
                value_max_grad_norm,
            } => {
                let value_layers =
                    &[&[self.observation_size][..], &value_layers[..], &[1]].concat();
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
        };
        Ok(learning_module)
    }
}
