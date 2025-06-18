use super::RolloutMode;
use super::{Env, EnvPool};
use crate::env::run_rollout;
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::Result;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

pub struct VecEnv<E: Env + Sync> {
    envs: Vec<E>,
    buffers: Vec<RolloutBuffer>,
}

impl<E: Env + Sync> EnvPool for VecEnv<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        self.buffers
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(idx, buffer)| {
                let env = &self.envs[idx];
                run_rollout(distribution, env, rollout_mode, buffer, None)
            })?;
        Ok(self.buffers.clone())
    }
}
