use candle_core::Tensor;
use std::marker::PhantomData;

use crate::{
    buffers::episode_bound_buffer::{StateBuffer, StepBoundBuffer},
    distributions::Distribution,
    env::{Env, RolloutMode},
    env_pools::EnvHolder,
    numeric::Buffer,
    utils::rollout_buffer::RolloutBuffer,
};

pub trait SequntialStepBoundHooks {
    type E: Env;

    fn process_last_step(&self, buffers: &mut Vec<StateBuffer<Self::E>>);

    // TODO: this should change
    fn post_process_hook(&self) {}
}

pub struct DefaultStepBoundHook<E: Env> {
    env: PhantomData<E>,
}

impl<E: Env> SequntialStepBoundHooks for DefaultStepBoundHook<E> {
    type E = E;

    fn process_last_step(&self, buffers: &mut Vec<StateBuffer<Self::E>>) {}

    fn post_process_hook(&self) {}
}

// TODO: we should make this generic
// This needs to be called fixed size sequential rollout holder
// fixed size -> no episode bound
pub struct VecEnvHolder2<E: Env, H: SequntialStepBoundHooks<E = E>> {
    pub step_bound: usize,
    pub buffers: Vec<StepBoundBuffer<E>>,
    pub hooks: H,
}

impl<E: Env<Tensor = Buffer>> VecEnvHolder2<E, DefaultStepBoundHook<E>> {
    pub fn new(step_bound: usize, envs: Vec<E>) -> Self {
        let buffers = envs
            .into_iter()
            .map(|env| StepBoundBuffer::new(env, step_bound))
            .collect();
        Self {
            step_bound,
            buffers,
            hooks: DefaultStepBoundHook { env: PhantomData },
        }
    }
}

impl<E: Env<Tensor = Buffer>, H: SequntialStepBoundHooks<E = E>> EnvHolder for VecEnvHolder2<E, H> {
    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    fn sequential_rollout<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn super::SequentialVecEnvHooks,
    ) -> candle_core::Result<Vec<RolloutBuffer>> {
        let mut steps_taken = 0;
        while steps_taken < self.step_bound {
            let mut state_buffers = vec![];
            for buf in self.buffers.iter_mut() {
                buf.step(distr);
                state_buffers.push(buf.move_buffer());
            }
            self.hooks.process_last_step(&mut state_buffers);
            for (buf, state_buffer) in self.buffers.iter_mut().zip(state_buffers) {
                buf.set_buffer(state_buffer);
            }
            steps_taken += self.buffers.len();
        }
        let steps_per_environment = self.step_bound / self.num_envs();
        Ok(self
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
        for buf in self.buffers.iter_mut() {
            for _ in 0..steps_per_environment {
                buf.step(distr);
            }
        }
        Ok(self
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
