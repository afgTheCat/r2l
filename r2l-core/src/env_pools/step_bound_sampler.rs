use crate::{
    distributions::Distribution,
    env::{Env, Sampler},
    env_pools::{
        fixed_size_env_pools::FixedSizeEnvPool, vector_env_holder2::SequntialStepBoundHooks,
    },
    numeric::Buffer,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;

pub struct StepBoundAsyncSampler<E: Env<Tensor = Buffer>, P: FixedSizeEnvPool<Env = E>> {
    pub step_bound: usize,
    pub env_pool: P,
}

impl<E: Env<Tensor = Buffer>, P: FixedSizeEnvPool<Env = E>> Sampler
    for StepBoundAsyncSampler<E, P>
{
    fn collect_rollouts<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distribution: &D,
    ) -> candle_core::Result<Vec<RolloutBuffer>> {
        let num_envs = self.env_pool.num_envs();
        let steps = self.step_bound / num_envs;
        self.env_pool.run_rollouts(distribution, steps);
        Ok(self.env_pool.to_rollout_buffers(steps))
    }
}

pub struct StepBoundSequentialSampler<
    E: Env<Tensor = Buffer>,
    P: FixedSizeEnvPool<Env = E>,
    H: SequntialStepBoundHooks<E = E>,
> {
    pub step_bound: usize,
    pub env_pool: P,
    pub hooks: H,
}

impl<E: Env<Tensor = Buffer>, P: FixedSizeEnvPool<Env = E>, H: SequntialStepBoundHooks<E = E>>
    Sampler for StepBoundSequentialSampler<E, P, H>
{
    fn collect_rollouts<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distribution: &D,
    ) -> candle_core::Result<Vec<RolloutBuffer>> {
        let mut steps_taken = 0;
        let num_envs = self.env_pool.num_envs();
        while steps_taken < self.step_bound {
            let mut state_buffers = self.env_pool.single_step_and_collect(distribution);
            self.hooks.process_last_step(&mut state_buffers);
            self.env_pool.set_buffers(state_buffers);
            steps_taken += num_envs;
        }
        let steps_per_environment = self.step_bound / num_envs;
        Ok(self.env_pool.to_rollout_buffers(steps_per_environment))
    }
}
