use candle_core::{DType, Device};
use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::on_policy_algorithms::a2c::{A2C, A2CParams};
use r2l_candle::learning_module::R2lCandleLearningModule;

use crate::{
    agents::{
        a2c::{BurnA2C, CandleA2C},
    },
    builders::{
        agent::AgentBuilder,
        learning_module::{LearningModuleBuilder, LearningModuleType},
        policy_distribution::{ActionSpaceType, DistributionType, PolicyDistributionBuilder},
        a2c::hook::DefaultA2CHookBuilder,
    },
    BurnBackend,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct A2CBurnBackend;

#[derive(Debug, Clone)]
pub struct A2CCandleBackend {
    pub device: Device,
}

pub struct A2CAgentBuilder<M> {
    pub a2c_params: A2CParams,
    pub distribution_builder: PolicyDistributionBuilder,
    pub hook_builder: DefaultA2CHookBuilder,
    pub actor_critic_type: LearningModuleBuilder,
    pub backend: M,
}

pub type A2CBurnLearningModuleBuilder = A2CAgentBuilder<A2CBurnBackend>;
pub type A2CCandleLearningModuleBuilder = A2CAgentBuilder<A2CCandleBackend>;

impl Default for A2CAgentBuilder<A2CBurnBackend> {
    fn default() -> Self {
        Self {
            hook_builder: DefaultA2CHookBuilder,
            a2c_params: A2CParams::default(),
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
            backend: A2CBurnBackend,
        }
    }
}

impl Default for A2CAgentBuilder<A2CCandleBackend> {
    fn default() -> Self {
        Self {
            hook_builder: DefaultA2CHookBuilder,
            a2c_params: A2CParams::default(),
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
            backend: A2CCandleBackend {
                device: Device::Cpu,
            },
        }
    }
}

impl<M> A2CAgentBuilder<M> {
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
    ) -> anyhow::Result<CandleA2C> {
        let lm = self.build_candle_lm(observation_size, action_size, action_space, &device)?;
        let hooks = self.hook_builder.build();
        let params = self.a2c_params;
        Ok(CandleA2C(A2C { lm, hooks, params }))
    }

    fn build_burn_agent(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<BurnA2C<BurnBackend>> {
        let distr = self.distribution_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let lm = self.actor_critic_type.build_burn(observation_size, distr);
        let hooks = self.hook_builder.build();
        let params = self.a2c_params;
        Ok(BurnA2C(A2C { lm, hooks, params }))
    }
}

impl A2CAgentBuilder<A2CCandleBackend> {
    pub fn with_candle_device(mut self, device: Device) -> Self {
        self.backend = A2CCandleBackend { device };
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
        self.build_burn_agent(observation_size, action_size, action_space)
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
        self.build_candle_with_device(observation_size, action_size, action_space, device)
    }
}
