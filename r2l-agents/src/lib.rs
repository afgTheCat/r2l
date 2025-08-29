pub mod a2c;
pub mod ppo;
pub mod vpg;

use candle_core::Tensor;
use r2l_core::{
    agents::Agent, distributions::DistributionKind, policies::learning_modules::LearningModuleKind,
    utils::rollout_buffer::RolloutBuffer,
};

use crate::{a2c::A2C, ppo::PPO, vpg::VPG};

pub enum AgentKind {
    A2C(A2C<DistributionKind, LearningModuleKind>),
    PPO(PPO<DistributionKind, LearningModuleKind>),
    VPG(VPG<DistributionKind, LearningModuleKind>),
}

impl Agent for AgentKind {
    type Dist = DistributionKind;

    fn learn(&mut self, rollouts: Vec<RolloutBuffer<Tensor>>) -> candle_core::Result<()> {
        match self {
            Self::A2C(a2c) => a2c.learn(rollouts),
            Self::PPO(ppo) => ppo.learn(rollouts),
            Self::VPG(vpg) => vpg.learn(rollouts),
        }
    }

    fn distribution(&self) -> &Self::Dist {
        match self {
            Self::A2C(a2c) => a2c.distribution(),
            Self::PPO(ppo) => ppo.distribution(),
            Self::VPG(vpg) => vpg.distribution(),
        }
    }
}
