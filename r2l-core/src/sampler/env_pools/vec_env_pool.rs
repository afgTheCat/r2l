use crate::{
    distributions::Policy,
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

    pub fn set_distr<D: Policy<Tensor = E::Tensor> + Clone>(&mut self, distr: D) {
        for buffer in self.buffers.iter_mut() {
            let distr: Box<dyn Policy<Tensor = E::Tensor>> = Box::new(distr.clone());
            buffer.set_distr(Some(distr));
        }
    }

    pub fn step2(&mut self, steps: usize) {
        for buf in self.buffers.iter_mut() {
            buf.step_n2(steps);
        }
    }
}

impl<E: Env> FixedSizeEnvPool for FixedSizeVecEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    fn step_n<D: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        for buffer in self.buffers.iter_mut() {
            let distr: Option<Box<dyn Policy<Tensor = E::Tensor>>> = Some(Box::new(distr.clone()));
            buffer.set_distr(distr);
            buffer.step_n2(steps);
        }
        self.buffers
            .iter_mut()
            .map(|buff| buff.take_rollout_buffer())
            .collect()
    }

    fn step_take_buffers(&mut self) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        for buffer in self.buffers.iter_mut() {
            buffer.step_n2(1);
        }
        self.buffers
            .iter_mut()
            .map(|buf| buf.move_buffer())
            .collect()
    }

    fn set_distr<D: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(&mut self, distr: D) {
        for buffer in self.buffers.iter_mut() {
            buffer.set_distr(Some(Box::new(distr.clone())));
        }
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        for (trajectory_buf, buf) in self.buffers.iter_mut().zip(buffers) {
            trajectory_buf.set_buffer(buf);
        }
    }

    fn take_rollout_buffers(&mut self) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        self.buffers
            .iter_mut()
            .map(|buff| buff.take_rollout_buffer())
            .collect()
    }
}

pub struct VariableSizedVecEnvPool<E: Env> {
    pub buffers: Vec<VariableSizedTrajectoryBuffer<E>>,
}

impl<E: Env> VariableSizedEnvPool for VariableSizedVecEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    fn step_with_episode_bound<D: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        for buffer in self.buffers.iter_mut() {
            let distr: Option<Box<dyn Policy<Tensor = E::Tensor>>> = Some(Box::new(distr.clone()));
            buffer.set_distr(distr);
            buffer.step_with_epiosde_bound2(steps);
        }
        self.buffers
            .iter_mut()
            .map(|buff| buff.take_rollout_buffer())
            .collect()
    }
}
