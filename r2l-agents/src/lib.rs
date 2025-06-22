pub mod a2c;
pub mod ppo;
pub mod vpg;

use crate::{a2c::A2C, ppo::PPO, vpg::VPG};
use candle_core::Result;
use r2l_core::{agents::Agent, policies::PolicyKind, utils::rollout_buffer::RolloutBuffer};

pub enum AgentKind {
    A2C(A2C<PolicyKind>),
    PPO(PPO<PolicyKind>),
    VPG(VPG<PolicyKind>),
}

// TODO: finish this
impl Agent for AgentKind {
    type Policy = PolicyKind;

    fn policy(&self) -> &Self::Policy {
        match &self {
            Self::A2C(a2c) => a2c.policy(),
            Self::PPO(ppo) => ppo.policy(),
            Self::VPG(vpg) => vpg.policy(),
        }
    }

    fn learn(&mut self, rollouts: Vec<RolloutBuffer>) -> Result<()> {
        match self {
            Self::A2C(a2c) => a2c.learn(rollouts),
            Self::PPO(ppo) => ppo.learn(rollouts),
            Self::VPG(vpg) => vpg.learn(rollouts),
        }
    }
}
