pub mod ppo;

use crate::builders::distribution::ActionSpaceType;
use r2l_core::agents::Agent;

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
