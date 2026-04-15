use r2l_core::on_policy::Agent;

use crate::builders::policy_distribution::ActionSpaceType;

// TODO: I am not even sure if we need this trait
pub trait AgentBuilder {
    type Agent: Agent;

    // TODO: This API is heavily in progress
    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent>;
}
