use crate::{
    distributions::Distribution,
    env::Env,
    numeric::Buffer,
    sampler::{
        env_pools::{FixedSizeEnvPool, VariableSizedEnvPool},
        trajectory_buffers::{
            fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
            variable_size_buffer::{VariableSizedStateBuffer, VariableSizedTrajectoryBuffer},
        },
    },
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;

pub struct FixedSizeVecEnvPool<E: Env> {
    pub buffers: Vec<FixedSizeTrajectoryBuffer<E>>,
}

impl<E: Env<Tensor = Buffer>> FixedSizeEnvPool for FixedSizeVecEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    // TODO: a more flexible environment would be one that can take a function
    fn single_step<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<E>> {
        let mut state_buffers = vec![];
        for buf in self.buffers.iter_mut() {
            buf.step(distr);
            state_buffers.push(buf.move_buffer());
        }
        state_buffers
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<E>>) {
        for (buf, state_buffer) in self.buffers.iter_mut().zip(buffers) {
            buf.set_buffer(state_buffer);
        }
    }

    fn to_rollout_buffers(&mut self, steps_per_environment: usize) -> Vec<RolloutBuffer> {
        self.buffers
            .iter_mut()
            .map(|buff| {
                buff.buffer
                    .as_mut()
                    .unwrap()
                    .to_rollout_buffer(steps_per_environment)
            })
            .collect()
    }

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        for buf in self.buffers.iter_mut() {
            for _ in 0..steps {
                buf.step(distr);
            }
        }
    }
}

pub struct VariableSizedVecEnvPool<E: Env> {
    pub buffers: Vec<VariableSizedTrajectoryBuffer<E>>,
}

impl<E: Env<Tensor = Buffer>> VariableSizedEnvPool for VariableSizedVecEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    fn single_step<D: Distribution<Tensor = Tensor>>(
        &mut self,
        env_idx: &[usize],
        distr: &D,
    ) -> Vec<VariableSizedStateBuffer<Self::Env>> {
        todo!()
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer> {
        todo!()
    }
}
