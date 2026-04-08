use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};

use crate::{
    agents::{
        AgentBuilder,
        ppo::{
            BurnOrCandlePPO,
            burn::{BurnBackend, BurnPPO},
            candle::{CandlePPO, R2lCandleLearningModule},
        },
    },
    builders::{
        distribution::{ActionSpaceType, DistributionBuilder, DistributionType},
        learning_module::{LearningModuleBuilder, LearningModuleType},
    },
    hooks::ppo::StandardPPOHookBuilder,
};

#[derive(Debug, Clone)]
pub enum PPOBackend {
    Burn,
    Candle(Device),
}

impl Default for PPOBackend {
    fn default() -> Self {
        Self::Burn
    }
}

pub struct UnifiedPPOLearningModuleBuilder {
    pub ppo_params: PPOParams,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: StandardPPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub backend: PPOBackend,
}

impl Default for UnifiedPPOLearningModuleBuilder {
    fn default() -> Self {
        Self {
            hook_builder: StandardPPOHookBuilder::default(),
            ppo_params: PPOParams::default(),
            distribution_builder: DistributionBuilder {
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
            backend: PPOBackend::default(),
        }
    }
}

impl UnifiedPPOLearningModuleBuilder {
    pub fn with_backend(mut self, backend: PPOBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_candle_device(mut self, device: Device) -> Self {
        self.backend = PPOBackend::Candle(device);
        self
    }

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

    fn build_candle(
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

    fn build_burn(
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
        let lm = self.actor_critic_type.build_burn::<BurnBackend, _>(distr);
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(BurnPPO(PPO { lm, hooks, params }))
    }
}

impl AgentBuilder for UnifiedPPOLearningModuleBuilder {
    type Agent = BurnOrCandlePPO;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        match self.backend.clone() {
            PPOBackend::Burn => Ok(BurnOrCandlePPO::Burn(self.build_burn(
                observation_size,
                action_size,
                action_space,
            )?)),
            PPOBackend::Candle(device) => Ok(BurnOrCandlePPO::Candle(self.build_candle(
                observation_size,
                action_size,
                action_space,
                device,
            )?)),
        }
    }
}
