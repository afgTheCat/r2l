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

pub type A2CBurnAgentBuilder = A2CAgentBuilder<A2CBurnBackend>;
pub type A2CCandleAgentBuilder = A2CAgentBuilder<A2CCandleBackend>;

impl A2CCandleAgentBuilder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultA2CHookBuilder::new(n_envs),
            a2c_params: A2CParams::default(),
            learning_module_builder: LearningModuleBuilder {
                policy_hidden_layers: vec![64, 64],
                learning_module_type: LearningModuleType::Joint {
                    value_hidden_layers: vec![64, 64],
                    max_grad_norm: None,
                },
                params: ParamsAdamW {
                    lr: 3e-4,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-5,
                    weight_decay: 1e-4,
                },
            },
            backend: A2CCandleBackend {
                device: Device::Cpu,
            },
        }
    }
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
