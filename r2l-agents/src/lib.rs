pub mod a2c;
pub mod ppo;
pub mod vpg;

use r2l_core::{distributions::DistributionKind, policies::learning_modules::LearningModuleKind};

use crate::{a2c::a2c3::A2C3, ppo::ppo3::PPO3, vpg::vpg3::VPG3};
// use candle_core::Result;
// use r2l_core::{agents::Agent, policies::PolicyKind, utils::rollout_buffer::RolloutBuffer};

pub enum AgentKind {
    A2C(A2C3<DistributionKind, LearningModuleKind>),
    PPO(PPO3<DistributionKind, LearningModuleKind>),
    VPG(VPG3<DistributionKind, LearningModuleKind>),
}

// TODO: finish this
// impl Agent for AgentKind {
//     type Policy = PolicyKind;
//
//     fn policy(&self) -> &Self::Policy {
//         match &self {
//             Self::A2C(a2c) => a2c.policy(),
//             Self::PPO(ppo) => ppo.policy(),
//             Self::PPO2(ppo) => ppo.policy(),
//             Self::VPG(vpg) => vpg.policy(),
//         }
//     }
//
//     fn learn(&mut self, rollouts: Vec<RolloutBuffer>) -> Result<()> {
//         match self {
//             Self::A2C(a2c) => a2c.learn(rollouts),
//             Self::PPO(ppo) => ppo.learn(rollouts),
//             Self::PPO2(ppo) => ppo.learn(rollouts),
//             Self::VPG(vpg) => vpg.learn(rollouts),
//         }
//     }
// }
