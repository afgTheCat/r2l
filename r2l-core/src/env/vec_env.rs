use super::RolloutMode;
use super::{Env, EnvPool, run_rollout};
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::Result;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

pub struct VecEnv<E: Env + Sync> {
    envs: Vec<E>,
    rollout_mode: RolloutMode,
}

impl<E: Env + Sync> EnvPool for VecEnv<E> {
    fn collect_rollouts<D: Distribution>(&self, distribution: &D) -> Result<Vec<RolloutBuffer>> {
        (0..self.envs.len())
            .into_par_iter()
            .map(|env_idx| run_rollout(distribution, &self.envs[env_idx], self.rollout_mode))
            .collect::<Result<Vec<_>>>()
    }
}
