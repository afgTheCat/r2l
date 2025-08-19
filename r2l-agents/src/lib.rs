pub mod a2c;
pub mod ppo;
pub mod vpg;

use r2l_core::{
    agents::Agent2, distributions::DistributionKind,
    policies::learning_modules::LearningModuleKind, utils::rollout_buffer::RolloutBuffer,
};

use crate::{a2c::a2c3::A2C3, ppo::ppo3::PPO3, vpg::vpg3::VPG3};
// use candle_core::Result;
// use r2l_core::{agents::Agent, policies::PolicyKind, utils::rollout_buffer::RolloutBuffer};

pub enum AgentKind {
    A2C(A2C3<DistributionKind, LearningModuleKind>),
    PPO(PPO3<DistributionKind, LearningModuleKind>),
    VPG(VPG3<DistributionKind, LearningModuleKind>),
}

impl Agent2 for AgentKind {
    type Dist = DistributionKind;

    fn learn(&mut self, rollouts: Vec<RolloutBuffer>) -> candle_core::Result<()> {
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
