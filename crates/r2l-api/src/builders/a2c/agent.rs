use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::a2c::{A2C, A2CParams};
use r2l_core::env::ActionSpaceType;

use crate::{
    BurnBackend,
    agents::a2c::{BurnA2C, CandleA2C},
    builders::{
        a2c::hook::DefaultA2CHookBuilder,
        agent::AgentBuilder,
        learning_module::{LearningModuleBuilder, LearningModuleType},
    },
    hooks::a2c::A2CStats,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct A2CBurnBackend;

#[derive(Debug, Clone)]
pub struct A2CCandleBackend {
    pub device: Device,
}

pub struct A2CAgentBuilder<M> {
    pub a2c_params: A2CParams,
    pub hook_builder: DefaultA2CHookBuilder,
    pub learning_module_builder: LearningModuleBuilder,
    pub backend: M,
}

impl<M> A2CAgentBuilder<M> {
    fn build_candle(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: Device,
    ) -> anyhow::Result<CandleA2C> {
        let lm = self.learning_module_builder.build_candle(
            observation_size,
            action_size,
            action_space,
            &device,
        )?;
        let hooks = self.hook_builder.build();
        let params = self.a2c_params;
        Ok(CandleA2C(A2C { lm, hooks, params }))
    }

    fn build_burn(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<BurnA2C<BurnBackend>> {
        let lm = self.learning_module_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let hooks = self.hook_builder.build();
        let params = self.a2c_params;
        Ok(BurnA2C(A2C { lm, hooks, params }))
    }

    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.hook_builder = self
            .hook_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.hook_builder = self.hook_builder.with_entropy_coeff(entropy_coeff);
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_vf_coeff(vf_coeff);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.hook_builder = self.hook_builder.with_tx(tx);
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.a2c_params.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.a2c_params.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.a2c_params.sample_size = sample_size;
        self
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

pub type A2CBurnAgentBuilder = A2CAgentBuilder<A2CBurnBackend>;
pub type A2CCandleAgentBuilder = A2CAgentBuilder<A2CCandleBackend>;

impl A2CCandleAgentBuilder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultA2CHookBuilder::new(n_envs),
            a2c_params: A2CParams::default(),
            learning_module_builder: LearningModuleBuilder {
                policy_hidden_layers: vec![64, 64],
                value_hidden_layers: vec![64, 64],
                learning_module_type: LearningModuleType::Joint {
                    max_grad_norm: None,
                    params: ParamsAdamW {
                        lr: 3e-4,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-5,
                        weight_decay: 1e-4,
                    },
                },
            },
            backend: A2CCandleBackend {
                device: Device::Cpu,
            },
        }
    }
}

impl A2CAgentBuilder<A2CCandleBackend> {
    pub fn with_candle(mut self, device: Device) -> Self {
        self.backend = A2CCandleBackend { device };
        self
    }

    pub fn with_burn(self) -> A2CAgentBuilder<A2CBurnBackend> {
        let A2CAgentBuilder {
            a2c_params,
            hook_builder,
            learning_module_builder,
            ..
        } = self;
        A2CAgentBuilder {
            a2c_params,
            hook_builder,
            learning_module_builder,
            backend: A2CBurnBackend,
        }
    }
}

impl A2CAgentBuilder<A2CBurnBackend> {
    pub fn with_candle(self, device: Device) -> A2CAgentBuilder<A2CCandleBackend> {
        let A2CAgentBuilder {
            a2c_params,
            hook_builder,
            learning_module_builder,
            ..
        } = self;
        A2CAgentBuilder {
            a2c_params,
            hook_builder,
            learning_module_builder,
            backend: A2CCandleBackend { device },
        }
    }

    pub fn with_burn(self) -> Self {
        self
    }
}

impl AgentBuilder for A2CAgentBuilder<A2CBurnBackend> {
    type Agent = BurnA2C<BurnBackend>;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        self.build_burn(observation_size, action_size, action_space)
    }
}

impl AgentBuilder for A2CAgentBuilder<A2CCandleBackend> {
    type Agent = CandleA2C;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        let device = self.backend.device.clone();
        self.build_candle(observation_size, action_size, action_space, device)
    }
}
