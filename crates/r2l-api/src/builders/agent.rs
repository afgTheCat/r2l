use r2l_core::agents::Agent;
use crate::builders::policy_distribution::ActionSpaceType;

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
