use burn::{prelude::Backend, tensor::backend::AutodiffBackend};
use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_burn::learning_module::PolicyValueModuleKind as BurnPolicyValueModule;
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
        todo!()
    }
}
