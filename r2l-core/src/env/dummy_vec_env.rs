use super::{Env, EnvPool, RolloutMode, run_rollout};
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::Result;

pub struct DummyVecEnv<E: Env> {
    pub n_env: usize,
    pub env: E,
    pub rollout_mode: RolloutMode,
}

impl<E: Env> EnvPool for DummyVecEnv<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
    ) -> Result<Vec<RolloutBuffer>> {
        (0..self.n_env)
            .map(|_| run_rollout(distribution, &self.env, self.rollout_mode))
            .collect::<Result<Vec<_>>>()
    }
}
