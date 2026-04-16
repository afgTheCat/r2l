use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};
use r2l_candle::learning_module::PolicyValueModule as CandlePolicyValueModule;

use crate::{
    BurnBackend,
    agents::ppo::{BurnPPO, CandlePPO},
    builders::{
        agent::AgentBuilder,
        learning_module::{LearningModuleBuilder, LearningModuleType},
        policy_builder::{ActionSpaceType, DistributionType, PolicyBuilder},
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
    pub policy_builder: PolicyBuilder,
    pub hook_builder: DefaultPPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub backend: M,
}

pub type BurnPPOAgentBuilder = PPOAgentBuilder<PPOBurnBackend>;
pub type CandlePPOAgentBuilder = PPOAgentBuilder<PPOCandleBackend>;

impl<M> PPOAgentBuilder<M> {
    fn build_candle_module(
        &self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: &Device,
    ) -> anyhow::Result<CandlePolicyValueModule> {
        let policy_varmap = VarMap::new();
        let policy_var_builder = VarBuilder::from_varmap(&policy_varmap, DType::F32, device);
        let policy = self.policy_builder.build_candle(
            &policy_var_builder,
            device,
            observation_size,
            action_size,
            action_space,
        )?;
        let (value_function, learning_module) = self.actor_critic_type.build_candle(
            policy_varmap,
            policy_var_builder,
            observation_size,
            device,
        )?;
        Ok(CandlePolicyValueModule {
            policy,
            optimizer: learning_module,
            value_function,
            device: device.clone(),
        })
    }

    fn build_candle(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
        device: Device,
    ) -> anyhow::Result<CandlePPO> {
        let lm = self.build_candle_module(observation_size, action_size, action_space, &device)?;
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
        let policy = self.policy_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let lm = self.actor_critic_type.build_burn(observation_size, policy);
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(BurnPPO(PPO { lm, hooks, params }))
    }
}

impl PPOAgentBuilder<PPOCandleBackend> {
    pub fn with_burn(self) -> PPOAgentBuilder<PPOBurnBackend> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            policy_builder: self.policy_builder,
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
            policy_builder: self.policy_builder,
            hook_builder: self.hook_builder,
            actor_critic_type: self.actor_critic_type,
            backend: PPOCandleBackend { device },
        }
    }

    pub fn with_burn(self) -> PPOAgentBuilder<PPOBurnBackend> {
        PPOAgentBuilder {
            ppo_params: self.ppo_params,
            policy_builder: self.policy_builder,
            hook_builder: self.hook_builder,
            actor_critic_type: self.actor_critic_type,
            backend: PPOBurnBackend,
        }
    }
}

impl CandlePPOAgentBuilder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultPPOHookBuilder::new(n_envs),
            ppo_params: PPOParams::default(),
            policy_builder: PolicyBuilder {
                hidden_layers: vec![64, 64],
                distribution_type: DistributionType::Dynamic,
            },
            actor_critic_type: LearningModuleBuilder {
                learning_module_type: LearningModuleType::Joint {
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
