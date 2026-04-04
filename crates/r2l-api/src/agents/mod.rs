pub mod burn_ppo;
pub mod candle_ppo;
pub mod candle_ppo2;

use crate::{
    agents::candle_ppo2::{DefaultPPO, R2lCandleLearningModule},
    builders::{
        distribution::{ActionSpaceType, DistributionBuilder, DistributionType},
        learning_module::{LearningModuleBuilder, LearningModuleType},
    },
    hooks::ppo::PPOHookBuilder,
};
use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::ppo2::{NewPPO, NewPPOParams};
use r2l_core::agents::Agent;

// NOTE: experimantally implementing it here. in the future this should not depend on candle
pub struct PPOCandleAgentBuilder {
    pub device: Device,
    pub distribution_builder: DistributionBuilder,
    pub hook_builder: PPOHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub ppo_params: NewPPOParams,
}

impl Default for PPOCandleAgentBuilder {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
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
            hook_builder: PPOHookBuilder::default(),
            ppo_params: NewPPOParams::default(),
        }
    }
}

impl PPOCandleAgentBuilder {
    fn build_lm(
        &mut self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<R2lCandleLearningModule> {
        let distribution_varmap = VarMap::new();
        let distr_var_builder =
            VarBuilder::from_varmap(&distribution_varmap, DType::F32, &self.device);
        let policy = self.distribution_builder.build(
            &distr_var_builder,
            &self.device,
            observation_size,
            action_size,
            action_space,
        )?;
        let (value_function, learning_module) = self.actor_critic_type.build(
            distribution_varmap,
            distr_var_builder,
            observation_size,
            &self.device,
        )?;
        let learning_module = R2lCandleLearningModule {
            policy,
            actor_critic: learning_module,
            value_function,
        };
        Ok(learning_module)
    }
}

impl AgentBuilder for PPOCandleAgentBuilder {
    type Agent = DefaultPPO;

    fn build(
        mut self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        let lm = self.build_lm(observation_size, action_size, action_space)?;
        let hooks = self.hook_builder.build();
        let params = self.ppo_params;
        Ok(DefaultPPO(NewPPO { lm, hooks, params }))
    }
}

pub trait AgentBuilder {
    type Agent: Agent;

    // TODO: the arguments to this funciton may not be final
    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent>;
}
