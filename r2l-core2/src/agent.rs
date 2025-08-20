use crate::{
    distributions::Distribution,
    env::{Action, Observation, SnapShot},
};

pub trait ValueFunction {
    type Observation: Observation;

    fn calculate_value(&self, observation: &Self::Observation) -> f32;
}

pub trait Agent {
    type Distribution: Distribution;

    /// Retrieve the underlying distribution throught the policy
    fn distribution(&self) -> Self::Distribution;

    /// Instruments learnging with the rollout buffers collected
    fn learn<O: Observation, A: Action>(&mut self, snapshots: Vec<SnapShot<O, A>>)
    where
        <Self::Distribution as Distribution>::Observation: From<O>,
        <Self::Distribution as Distribution>::Action: From<A>;
}

pub type AgentObs<A> = <<A as Agent>::Distribution as Distribution>::Observation;
pub type AgentAct<A> = <<A as Agent>::Distribution as Distribution>::Action;
