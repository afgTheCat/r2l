use crate::env::{Action, Observation, SnapShot};

// This rollout buffer thing needs a bit of rethinking
pub struct RolloutBufferV2<O: Observation, A: Action> {
    pub snapshots: Vec<SnapShot<O, A>>, // snapshots
    pub last_state: Option<O>,          // last state that we ended up in
}

impl<O: Observation, A: Action> RolloutBufferV2<O, A> {
    pub fn push_snapshot(&mut self, snapshot: SnapShot<O, A>) {
        self.snapshots.push(snapshot);
    }

    pub fn set_last_state(&mut self, last_state: O) {
        self.last_state = Some(last_state)
    }

    // TODO: this is the old code.
    // pub fn calculate_advantages_and_returns(
    //     &self,
    //     policy: &impl PolicyWithValueFunction,
    //     gamma: f32,
    //     lambda: f32,
    // ) -> Result<(Vec<f32>, Vec<f32>)> {
    //     let states = Tensor::stack(&self.states, 0)?;
    //     let values: Vec<f32> = policy.calculate_values(&states)?.to_vec1()?;
    //     let total_steps = self.rewards.len();
    //     let mut advantages: Vec<f32> = vec![0.; total_steps];
    //     let mut returns: Vec<f32> = vec![0.; total_steps];
    //     let mut last_gae_lam: f32 = 0.;
    //     for i in (0..total_steps).rev() {
    //         let next_non_terminal = if self.dones[i] {
    //             last_gae_lam = 0.;
    //             0f32
    //         } else {
    //             1.
    //         };
    //         let delta = self.rewards[i] + next_non_terminal * gamma * values[i + 1] - values[i];
    //         last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
    //         advantages[i] = last_gae_lam;
    //         returns[i] = last_gae_lam + values[i];
    //     }
    //     Ok((advantages, returns))
    // }
}
