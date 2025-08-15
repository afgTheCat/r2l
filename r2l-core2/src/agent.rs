use super::policies::Policy;
use crate::env::{Action, Observation, SnapShot};

pub trait Agent {
    type Obs: Observation;
    type Act: Action;
    type Policy: Policy<Obs = Self::Obs, Act = Self::Act>;

    /// Retrieves the underlying policy
    fn policy(&self) -> &Self::Policy;

    /// Retrieve the underlying distribution throught the policy
    fn distribution(&self) -> <Self::Policy as Policy>::Dist {
        self.policy().distribution()
    }

    /// Instruments learnging with the rollout buffers collected
    fn learn<O: Observation, A: Action>(&mut self, snapshots: Vec<SnapShot<O, A>>)
    where
        Self::Obs: From<O>,
        Self::Act: From<A>;
}
