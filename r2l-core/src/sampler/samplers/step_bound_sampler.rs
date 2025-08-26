use std::marker::PhantomData;

use crate::{
    distributions::Distribution,
    env::{Env, RolloutMode, Sampler},
    env_pools::{EnvHolder, SequentialVecEnvHooks},
    numeric::Buffer,
    sampler::{
        env_pools::{FixedSizeEnvPool, vec_env_pool::FixedSizeVecEnvPool},
        trajectory_buffers::fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
    },
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;

pub trait SequntialStepBoundHooks {
    type E: Env;

    fn process_last_step(&self, buffers: &mut Vec<FixedSizeStateBuffer<Self::E>>);

    // TODO: this should change
    fn post_process_hook(&self) {}
}

pub struct DefaultStepBoundHook<E: Env> {
    env: PhantomData<E>,
}

impl<E: Env> SequntialStepBoundHooks for DefaultStepBoundHook<E> {
    type E = E;

    fn process_last_step(&self, buffers: &mut Vec<FixedSizeStateBuffer<Self::E>>) {}

    fn post_process_hook(&self) {}
}

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
            let mut state_buffers = self.env_pool.single_step(distribution);
            self.hooks.process_last_step(&mut state_buffers);
            self.env_pool.set_buffers(state_buffers);
            steps_taken += num_envs;
        }
        let steps_per_environment = self.step_bound / num_envs;
        Ok(self.env_pool.to_rollout_buffers(steps_per_environment))
    }
}

// TODO: remove this!
pub type VecEnvHolder2<E, H> = StepBoundSequentialSampler<E, FixedSizeVecEnvPool<E>, H>;

impl<E: Env<Tensor = Buffer>> VecEnvHolder2<E, DefaultStepBoundHook<E>> {
    pub fn new(step_bound: usize, envs: Vec<E>) -> Self {
        let buffers = envs
            .into_iter()
            .map(|env| FixedSizeTrajectoryBuffer::new(env, step_bound))
            .collect();
        Self {
            step_bound,
            env_pool: FixedSizeVecEnvPool { buffers },
            hooks: DefaultStepBoundHook { env: PhantomData },
        }
    }
}

impl<E: Env<Tensor = Buffer>, H: SequntialStepBoundHooks<E = E>> EnvHolder for VecEnvHolder2<E, H> {
    fn num_envs(&self) -> usize {
        self.env_pool.num_envs()
    }

    fn sequential_rollout<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> candle_core::Result<Vec<RolloutBuffer>> {
        let mut steps_taken = 0;
        while steps_taken < self.step_bound {
            let mut state_buffers = vec![];
            for buf in self.env_pool.buffers.iter_mut() {
                buf.step(distr);
                state_buffers.push(buf.move_buffer());
            }
            self.hooks.process_last_step(&mut state_buffers);
            for (buf, state_buffer) in self.env_pool.buffers.iter_mut().zip(state_buffers) {
                buf.set_buffer(state_buffer);
            }
            steps_taken += self.env_pool.buffers.len();
        }
        let steps_per_environment = self.step_bound / self.num_envs();
        Ok(self
            .env_pool
            .buffers
            .iter_mut()
            .map(|buff| {
                buff.buffer
                    .as_mut()
                    .unwrap()
                    .to_rollout_buffer(steps_per_environment)
            })
            .collect())
    }

    fn async_rollout<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> candle_core::Result<Vec<RolloutBuffer>> {
        let steps_per_environment = self.step_bound / self.num_envs();
        for buf in self.env_pool.buffers.iter_mut() {
            for _ in 0..steps_per_environment {
                buf.step(distr);
            }
        }
        Ok(self
            .env_pool
            .buffers
            .iter_mut()
            .map(|buff| {
                buff.buffer
                    .as_mut()
                    .unwrap()
                    .to_rollout_buffer(steps_per_environment)
            })
            .collect())
    }
}
