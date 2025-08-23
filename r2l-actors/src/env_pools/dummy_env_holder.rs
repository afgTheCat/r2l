// This needs to hold N buffers. Each buffer should also tell you about the stopping condition?
// Or should that be the responsibility of the env pool? YES. If you fuck up the buffers it's your
// fault

use r2l_core2::{distributions::Distribution, env::Env};

use crate::{
    buffers::{episode_bound_buffer::EpisodeBoundBuffer, step_bound_buffer::StepBoundBuffer},
    env_pools::SequntialStepBoundHooks,
};

struct StepBoundDummyVecEnv<E: Env, H: SequntialStepBoundHooks<E::Obs, E::Act>> {
    step_bound: usize,
    buffers: Vec<StepBoundBuffer<E>>,
    hooks: H,
}

impl<E: Env, H: SequntialStepBoundHooks<E::Obs, E::Act>> StepBoundDummyVecEnv<E, H> {
    fn sequential_rollout(
        &mut self,
        distr: &impl Distribution<Observation = E::Obs, Action = E::Act>,
    ) {
        let mut steps_taken = 0;
        while steps_taken < self.step_bound {
            let snapshots: Vec<_> = self.buffers.iter_mut().map(|buf| buf.step(distr)).collect();
            let snapshots = self.hooks.process_snapshots(snapshots);
            for (i, snapshot) in snapshots.into_iter().enumerate() {
                self.buffers[i].push_snapshot(snapshot);
            }
            steps_taken += self.buffers.len();
        }
        // TODO: post rollout hook here
    }
}

// TODO: fully flesh this out
struct EpisodeBoundDummyVecEnv<E: Env> {
    episode_bound: usize,
    buffer: Vec<EpisodeBoundBuffer<E>>,
}
