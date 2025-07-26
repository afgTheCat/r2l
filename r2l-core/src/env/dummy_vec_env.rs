// TODO: we will need to integrate the Evaluator as a hook.
use super::{Env, EnvPool, RolloutMode};
use crate::{
    distributions::Distribution,
    env::{EnvironmentDescription, run_rollout},
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Result;

pub struct DummyVecEnv<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: Vec<E>,
    pub env_description: EnvironmentDescription,
}

impl<E: Env> EnvPool for DummyVecEnv<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        for (env_idx, buffer) in self.buffers.iter_mut().enumerate() {
            run_rollout(distribution, &mut self.env[env_idx], rollout_mode, buffer)?;
        }
        Ok(self.buffers.clone())
    }

    fn env_description(&self) -> EnvironmentDescription {
        self.env_description.clone()
    }

    fn num_env(&self) -> usize {
        self.env.len()
    }
}
