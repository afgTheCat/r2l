use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};
use r2l_core::env::ActionSpaceType;

use crate::{
    BurnBackend,
    agents::ppo::{BurnPPO, CandlePPO},
    builders::{
        agent::AgentBuilder,
        learning_module::{LearningModuleBuilder, LearningModuleType},
        ppo::hook::DefaultPPOHookBuilder,
    },
};

#[derive(Debug, Clone, Copy, Default)]
pub struct PPOBurnBackend;

#[derive(Debug, Clone)]
pub struct PPOCandleBackend {
    pub device: Device,
}

pub struct PPOAgentBuilder<M = PPOCandleBackend> {
    pub ppo_params: PPOParams,
    pub hook_builder: DefaultPPOHookBuilder,
    pub learning_module_builder: LearningModuleBuilder,
    pub backend: M,
}

pub type BurnPPOAgentBuilder = PPOAgentBuilder<PPOBurnBackend>;
pub type CandlePPOAgentBuilder = PPOAgentBuilder<PPOCandleBackend>;

impl<M> PPOAgentBuilder<M> {
    fn build_candle(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: Device,
    ) -> anyhow::Result<CandlePPO> {
        let lm = self.learning_module_builder.build_candle(
            observation_size,
            action_size,
            action_space,
            &device,
        )?;
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(CandlePPO(PPO { lm, hooks, params }))
    }

    fn build_burn(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<BurnPPO<BurnBackend>> {
        let lm = self.learning_module_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(BurnPPO(PPO { lm, hooks, params }))
    }
}

impl PPOAgentBuilder<PPOCandleBackend> {
    pub fn with_burn(self) -> PPOAgentBuilder<PPOBurnBackend> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            hook_builder: self.hook_builder,
            learning_module_builder: self.learning_module_builder,
            backend: PPOBurnBackend,
        }
    }

    pub fn with_candle(mut self, device: Device) -> PPOAgentBuilder<PPOCandleBackend> {
        self.backend = PPOCandleBackend { device };
        self
    }
}

impl PPOAgentBuilder<PPOBurnBackend> {
    pub fn with_candle(self, device: Device) -> PPOAgentBuilder<PPOCandleBackend> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            hook_builder: self.hook_builder,
            learning_module_builder: self.learning_module_builder,
            backend: PPOCandleBackend { device },
        }
    }

    pub fn with_burn(self) -> PPOAgentBuilder<PPOBurnBackend> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            hook_builder: self.hook_builder,
            learning_module_builder: self.learning_module_builder,
            backend: PPOBurnBackend,
        }
    }
}

impl CandlePPOAgentBuilder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultPPOHookBuilder::new(n_envs),
            ppo_params: PPOParams::default(),
            learning_module_builder: LearningModuleBuilder {
                policy_hidden_layers: vec![64, 64],
                value_hidden_layers: vec![64, 64],
                learning_module_type: LearningModuleType::Joint {
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
            backend: PPOCandleBackend {
                device: Device::Cpu,
            },
        }
    }
}

impl AgentBuilder for PPOAgentBuilder<PPOBurnBackend> {
    type Agent = BurnPPO<BurnBackend>;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        self.build_burn(observation_size, action_size, action_space)
    }
}

impl AgentBuilder for PPOAgentBuilder<PPOCandleBackend> {
    type Agent = CandlePPO;

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
