use crate::{
    distributions::Distribution,
    env::{Env, EnvironmentDescription},
    sampler::{
        env_pools::{FixedSizeEnvPool, VariableSizedEnvPool},
        trajectory_buffers::{
            fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
            variable_size_buffer::VariableSizedTrajectoryBuffer,
        },
    },
    utils::rollout_buffer::RolloutBuffer,
};

pub struct FixedSizeVecEnvPool<E: Env> {
    pub buffers: Vec<FixedSizeTrajectoryBuffer<E>>,
}

impl<E: Env> FixedSizeVecEnvPool<E> {
    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        self.buffers[0].env.env_description()
    }

    pub fn set_distr<D: Distribution<Tensor = E::Tensor> + Clone>(&mut self, distr: D) {
        for buffer in self.buffers.iter_mut() {
            let distr: Box<dyn Distribution<Tensor = E::Tensor>> = Box::new(distr.clone());
            buffer.set_distr(Some(distr));
        }
    }

    pub fn step2(&mut self, steps: usize) {
        for buf in self.buffers.iter_mut() {
            buf.step_n2(steps);
        }
    }

    pub fn step_take_buffers2(&mut self) -> Vec<FixedSizeStateBuffer<E>> {
        self.step2(1);
        self.buffers
            .iter_mut()
            .map(|buf| buf.move_buffer())
            .collect()
    }
}

impl<E: Env> FixedSizeEnvPool for FixedSizeVecEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    fn step<D: Distribution<Tensor = E::Tensor>>(&mut self, distr: &D, steps: usize) {
        for buf in self.buffers.iter_mut() {
            buf.step_n(distr, steps);
        }
    }

    fn step_take_buffers<D: Distribution<Tensor = E::Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        self.step(distr, 1);
        self.buffers
            .iter_mut()
            .map(|buf| buf.move_buffer())
            .collect()
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer<E::Tensor>> {
        self.buffers
            .iter_mut()
            .map(|buff| buff.to_rollout_buffer())
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

impl<E: Env> VariableSizedVecEnvPool<E> {
    fn step_with_epiosde_bound2<D: Distribution<Tensor = E::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) {
        for buffer in self.buffers.iter_mut() {
            let distr: Option<Box<dyn Distribution<Tensor = E::Tensor>>> =
                Some(Box::new(distr.clone()));
            buffer.set_distr(distr);
            buffer.step_with_epiosde_bound2(steps);
        }
    }
}

impl<E: Env> VariableSizedEnvPool for VariableSizedVecEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer<E::Tensor>> {
        self.buffers
            .iter_mut()
            .map(|buff| buff.take_rollout_buffer())
            .collect()
    }

    fn step_with_episode_bound<D: Distribution<Tensor = E::Tensor>>(
        &mut self,
        distr: &D,
        steps: usize,
    ) {
        for buffer in self.buffers.iter_mut() {
            buffer.step_with_epiosde_bound(distr, steps);
        }
    }
}
