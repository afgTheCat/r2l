use r2l_core2::{
    distributions::Distribution,
    env::{Action, Env, Observation},
    utils::rollout_buffers::RolloutBufferV2,
};

use crate::RolloutMode;

// The environment has to be associated here!
pub struct VecEnvHolder<E: Env> {
    pub envs: Vec<E>,
    pub buffers: Vec<RolloutBufferV2<E::Obs, E::Act>>,
}

impl<E: Env> VecEnvHolder<E> {
    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    // TODO: will probably simplify in the future
    fn sequential_rollout<O: Observation, A: Action, D: Distribution<O, A>>(
        &mut self,
        distr: D,
        rollout_mode: RolloutMode,
    ) {
    }

    // not really async in this case, only each environment does the thing after one another
    fn async_rollout(&mut self) {}
}
