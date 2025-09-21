pub mod burn_agents;
pub mod candle_agents;

use crate::candle_agents::{ModuleWithValueFunction, a2c::A2C, ppo::CandlePPO, vpg::VPG};
use anyhow::Result;
use candle_core::Tensor;
use r2l_candle_lm::{
    distributions::DistributionKind,
    learning_module2::PolicyValuesLosses,
    learning_module2::{DecoupledActorCriticLM2, ParalellActorCriticLM2, SequentialValueFunction},
};
use r2l_core::{
    agents::Agent, distributions::Policy, policies::LearningModule,
    utils::rollout_buffer::RolloutBuffer,
};

pub struct GenericLearningModuleWithValueFunction<
    P: Policy<Tensor = Tensor> + Clone,
    L: LearningModule<Losses = PolicyValuesLosses>,
> {
    pub policy: P,
    pub learning_module: L,
    pub value_function: SequentialValueFunction,
}

impl<P: Policy<Tensor = Tensor> + Clone, L: LearningModule<Losses = PolicyValuesLosses>>
    ModuleWithValueFunction for GenericLearningModuleWithValueFunction<P, L>
{
    type P = P;
    type L = L;
    type V = SequentialValueFunction;

    fn get_inference_policy(&self) -> Self::P {
        self.policy.clone()
    }

    fn get_policy_ref(&self) -> &Self::P {
        &self.policy
    }

    fn learning_module(&mut self) -> &mut Self::L {
        &mut self.learning_module
    }

    fn value_func(&self) -> &Self::V {
        &self.value_function
    }
}

pub enum ActorCriticKind {
    Decoupled(DecoupledActorCriticLM2),
    Paralell(ParalellActorCriticLM2),
}

impl ActorCriticKind {
    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Decoupled(decoupled) => decoupled.policy_learning_rate(),
            Self::Paralell(paralell) => paralell.policy_learning_rate(),
        }
    }
}

impl LearningModule for ActorCriticKind {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        match self {
            Self::Decoupled(lm) => lm.update(losses),
            Self::Paralell(lm) => lm.update(losses),
        }
    }
}

pub type LearningModuleKind =
    GenericLearningModuleWithValueFunction<DistributionKind, ActorCriticKind>;

pub enum AgentKind {
    A2C(A2C<LearningModuleKind>),
    PPO(CandlePPO<LearningModuleKind>),
    VPG(VPG<LearningModuleKind>),
}

impl Agent for AgentKind {
    type Policy = DistributionKind;

    fn learn(&mut self, rollouts: Vec<RolloutBuffer<Tensor>>) -> Result<()> {
        match self {
            Self::A2C(a2c) => a2c.learn(rollouts),
            Self::PPO(ppo) => ppo.learn(rollouts),
            Self::VPG(vpg) => vpg.learn(rollouts),
        }
    }

    fn policy(&self) -> Self::Policy {
        match self {
            Self::A2C(a2c) => a2c.policy(),
            Self::PPO(ppo) => ppo.policy(),
            Self::VPG(vpg) => vpg.policy(),
        }
    }
}
