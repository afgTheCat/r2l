use crate::{
    env::{Action, Observation, SnapShot},
    policies::PolicyWithValueFunction,
};

// This rollout buffer thing needs a bit of rethinking
// Another question is whether we want to deal with Observation and Action having different types
pub struct RolloutBufferV2<O: Observation, A: Action> {
    pub snapshots: Vec<SnapShot<O, A>>, // snapshots
    pub last_state: Option<O>,          // last state that we ended up in
}

// This is actually fine like this, the implementor can decide what the datatype should be
pub struct Batch<O: Observation, A: Action> {
    pub observations: Vec<O>,
    pub actions: Vec<A>,
    pub returns: Vec<f32>,
    pub advantages: Vec<f32>,
    pub logp_old: Vec<f32>,
}

impl<O: Observation, A: Action> RolloutBufferV2<O, A> {
    pub fn push_snapshot(&mut self, snapshot: SnapShot<O, A>) {
        self.snapshots.push(snapshot);
    }

    pub fn set_last_state(&mut self, last_state: O) {
        self.last_state = Some(last_state)
    }

    pub fn calculate_advantages_and_returns(
        &self,
        policy: &impl PolicyWithValueFunction<Obs = O>,
        gamma: f32,
        lambda: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let total_steps = self.snapshots.len();
        let Some(last_state) = self.last_state.clone() else {
            panic!("Last state should be set when calculating advantages and returns");
        };
        let values = self
            .snapshots
            .iter()
            .map(|s| policy.calculate_value(s.state.clone()))
            .chain(std::iter::once(policy.calculate_value(last_state)))
            .collect::<Vec<_>>();
        let mut advantages: Vec<f32> = vec![0.; total_steps];
        let mut returns: Vec<f32> = vec![0.; total_steps];
        let mut last_gae_lam: f32 = 0.;
        for i in (0..total_steps).rev() {
            let next_non_terminal = if self.snapshots[i].done() {
                last_gae_lam = 0.;
                0f32
            } else {
                1.
            };
            let delta =
                self.snapshots[i].reward + next_non_terminal * gamma * values[i + 1] - values[i];
            last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
            advantages[i] = last_gae_lam;
            returns[i] = last_gae_lam + values[i];
        }
        (advantages, returns)
    }

    pub fn get_batch(&mut self) -> Option<Batch<O, A>> {
        todo!()
    }
}
