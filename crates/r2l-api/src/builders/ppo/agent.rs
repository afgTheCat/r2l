use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};
use r2l_candle::learning_module::R2lCandleLearningModule;

use crate::{
    BurnBackend,
    agents::ppo::{BurnPPO, CandlePPO},
    builders::{
        agent::AgentBuilder,
        learning_module::{LearningModuleBuilder, LearningModuleType},
        policy_distribution::{ActionSpaceType, DistributionType, PolicyDistributionBuilder},
        ppo::hook::StandardPPOHookBuilder,
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
    pub distribution_builder: PolicyDistributionBuilder,
    pub hook_builder: StandardPPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub backend: M,
}

pub type PPOBurnLearningModuleBuilder = PPOAgentBuilder<PPOBurnBackend>;
pub type PPOCandleLearningModuleBuilder = PPOAgentBuilder<PPOCandleBackend>;

impl<M> PPOAgentBuilder<M> {
    fn build_candle_lm(
        &self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: &Device,
    ) -> anyhow::Result<R2lCandleLearningModule> {
        let distribution_varmap = VarMap::new();
        let distr_var_builder = VarBuilder::from_varmap(&distribution_varmap, DType::F32, device);
        let policy = self.distribution_builder.build_candle(
            &distr_var_builder,
            device,
            observation_size,
            action_size,
            action_space,
        )?;
        let (value_function, learning_module) = self.actor_critic_type.build_candle(
            distribution_varmap,
            distr_var_builder,
            observation_size,
            device,
        )?;
        Ok(R2lCandleLearningModule {
            policy,
            actor_critic: learning_module,
            value_function,
            device: device.clone(),
        })
    }

    fn build_candle_with_device(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: Device,
    ) -> anyhow::Result<CandlePPO> {
        let lm = self.build_candle_lm(observation_size, action_size, action_space, &device)?;
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(CandlePPO(PPO { lm, hooks, params }))
    }

    fn build_burn_agent(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<BurnPPO<BurnBackend>> {
        let distr = self.distribution_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let lm = self.actor_critic_type.build_burn(observation_size, distr);
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(BurnPPO(PPO { lm, hooks, params }))
    }
}

impl PPOAgentBuilder<PPOCandleBackend> {
    pub fn with_burn(self) -> PPOAgentBuilder<PPOBurnBackend> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            distribution_builder: self.distribution_builder,
            hook_builder: self.hook_builder,
            actor_critic_type: self.actor_critic_type,
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
            distribution_builder: self.distribution_builder,
            hook_builder: self.hook_builder,
            actor_critic_type: self.actor_critic_type,
            backend: PPOCandleBackend { device },
        }
    }

    pub fn with_burn(self) -> PPOAgentBuilder<PPOBurnBackend> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            distribution_builder: self.distribution_builder,
            hook_builder: self.hook_builder,
            actor_critic_type: self.actor_critic_type,
            backend: PPOBurnBackend,
        }
    }
}

impl Default for PPOAgentBuilder<PPOBurnBackend> {
    fn default() -> Self {
        Self {
            hook_builder: StandardPPOHookBuilder::default(),
            ppo_params: PPOParams::default(),
            distribution_builder: PolicyDistributionBuilder {
                hidden_layers: vec![64, 64],
                distribution_type: DistributionType::Dynamic,
            },
            actor_critic_type: LearningModuleBuilder {
                learning_module_type: LearningModuleType::Paralell {
                    value_layers: vec![64, 64],
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
            backend: PPOBurnBackend,
        }
    }
}

impl Default for PPOAgentBuilder<PPOCandleBackend> {
    fn default() -> Self {
        Self {
            hook_builder: StandardPPOHookBuilder::default(),
            ppo_params: PPOParams::default(),
            distribution_builder: PolicyDistributionBuilder {
                hidden_layers: vec![64, 64],
                distribution_type: DistributionType::Dynamic,
            },
            actor_critic_type: LearningModuleBuilder {
                learning_module_type: LearningModuleType::Paralell {
                    value_layers: vec![64, 64],
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
        self.build_burn_agent(observation_size, action_size, action_space)
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
        self.build_candle_with_device(observation_size, action_size, action_space, device)
    }
}
