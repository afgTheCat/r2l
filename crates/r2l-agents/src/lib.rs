pub mod burn_agents;
pub mod candle_agents;

use anyhow::Result;
use candle_core::Tensor;
use r2l_candle_lm::{distributions::DistributionKind, learning_module::LearningModuleKind};
use r2l_core::{agents::Agent, utils::rollout_buffer::RolloutBuffer};

use crate::candle_agents::{a2c::A2C, ppo::CandlePPO, vpg::VPG};

pub enum AgentKind {
    A2C(A2C<DistributionKind, LearningModuleKind>),
    PPO(CandlePPO<DistributionKind, LearningModuleKind>),
    VPG(VPG<DistributionKind, LearningModuleKind>),
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
