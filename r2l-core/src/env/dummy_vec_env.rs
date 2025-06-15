use super::{Env, EnvPool, RolloutMode};
use crate::{distributions::Distribution, env::run_rollout, utils::rollout_buffer::RolloutBuffer};
use candle_core::Result;

pub struct DummyVecEnv<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: E,
}

impl<E: Env> EnvPool for DummyVecEnv<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        for buffer in self.buffers.iter_mut() {
            run_rollout(distribution, &self.env, rollout_mode, buffer)?;
        }
        Ok(self.buffers.clone())
    }
}
