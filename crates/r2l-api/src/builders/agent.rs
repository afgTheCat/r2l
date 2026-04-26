use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_core::{env::ActionSpaceType, on_policy::algorithm::Agent};

use crate::builders::learning_module::{LearningModuleBuilder, LearningModuleType};

pub trait AgentBuilder {
    type Agent: Agent;

    // TODO: This API is heavily in progress
    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent>;
}

pub struct AgentBuilderStruct<Params, HookBuilder, Backend> {
    pub(crate) params: Params,
    pub(crate) hook_builder: HookBuilder,
    pub(crate) learning_module_builder: LearningModuleBuilder,
    pub(crate) backend: Backend,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BurnBackend;

#[derive(Debug, Clone)]
pub struct CandleBackend {
    pub device: Device,
}

impl<Params, HookBuilder, Backend> AgentBuilderStruct<Params, HookBuilder, Backend> {
    pub fn with_candle(
        self,
        device: Device,
    ) -> AgentBuilderStruct<Params, HookBuilder, CandleBackend> {
        let AgentBuilderStruct {
            params,
            hook_builder,
            learning_module_builder,
            ..
        } = self;
        AgentBuilderStruct {
            params,
            hook_builder,
            learning_module_builder,
            backend: CandleBackend { device },
        }
    }

    pub fn with_burn(self) -> AgentBuilderStruct<Params, HookBuilder, BurnBackend> {
        let AgentBuilderStruct {
            params,
            hook_builder,
            learning_module_builder,
            ..
        } = self;
        AgentBuilderStruct {
            params,
            hook_builder,
            learning_module_builder,
            backend: BurnBackend,
        }
    }

    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.learning_module_builder.policy_hidden_layers = policy_hidden_layers;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_lr(learning_rate);
        self
    }

    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_beta1(beta1);
        self
    }

    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_beta2(beta2);
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_epsilon(epsilon);
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_weight_decay(weight_decay);
        self
    }

    pub fn with_joint(mut self, max_grad_norm: Option<f32>, params: ParamsAdamW) -> Self {
        self.learning_module_builder.learning_module_type = LearningModuleType::Joint {
            max_grad_norm,
            params,
        };
        self
    }

    pub fn with_split(
        mut self,
        policy_max_grad_norm: Option<f32>,
        policy_params: ParamsAdamW,
        value_max_grad_norm: Option<f32>,
        value_params: ParamsAdamW,
    ) -> Self {
        self.learning_module_builder.learning_module_type = LearningModuleType::Split {
            policy_max_grad_norm,
            policy_params,
            value_max_grad_norm,
            value_params,
        };
        self
    }

    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.learning_module_builder.value_hidden_layers = value_hidden_layers;
        self
    }

    pub fn with_learning_module_type(mut self, learning_module_type: LearningModuleType) -> Self {
        self.learning_module_builder.learning_module_type = learning_module_type;
        self
    }
}
