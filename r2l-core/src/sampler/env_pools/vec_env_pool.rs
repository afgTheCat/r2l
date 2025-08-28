use crate::{
    distributions::Distribution,
    env::Env,
    numeric::Buffer,
    sampler::{
        env_pools::{FixedSizeEnvPool, VariableSizedEnvPool},
        trajectory_buffers::{
            fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
            variable_size_buffer::VariableSizedTrajectoryBuffer,
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

    fn step<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        for buf in self.buffers.iter_mut() {
            buf.step_n(distr, steps);
        }
    }

    fn step_take_buffers<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        self.step(distr, 1);
        self.buffers
            .iter_mut()
            .map(|buf| buf.move_buffer())
            .collect()
    }

    fn to_rollout_buffers(&self) -> Vec<RolloutBuffer> {
        self.buffers
            .iter()
            .map(|buff| {
                let buff = buff.buffer.as_ref().unwrap();
                buff.to_rollout_buffers2()
            })
            .collect()
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        for (trajectory_buf, buf) in self.buffers.iter_mut().zip(buffers) {
            trajectory_buf.set_buffer(buf);
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

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer> {
        self.buffers
            .iter()
            .map(|buff| buff.to_rollout_buffer())
            .collect()
    }

    fn step_with_episode_bound<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        steps: usize,
    ) {
        for buffer in self.buffers.iter_mut() {
            buffer.step_with_epiosde_bound(distr, steps);
        }
    }
}
