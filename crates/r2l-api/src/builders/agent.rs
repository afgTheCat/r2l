use r2l_core::{env::ActionSpaceType, on_policy::algorithm::Agent};

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
