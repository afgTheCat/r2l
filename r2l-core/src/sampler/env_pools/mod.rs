use crate::{
    distributions::Distribution,
    env::Env,
    numeric::Buffer,
    sampler::{
        env_pools::{
            thread_env_pool::{FixedSizeThreadEnvPool, VariableSizedThreadEnvPool},
            vec_env_pool::{FixedSizeVecEnvPool, VariableSizedVecEnvPool},
        },
        trajectory_buffers::fixed_size_buffer::FixedSizeStateBuffer,
    },
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;

pub mod thread_env_pool;
pub mod vec_env_pool;

pub trait FixedSizeEnvPool {
    type Env: Env<Tensor = Buffer>;

    fn num_envs(&self) -> usize;

    // step each environment steps time
    fn step<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize);

    // transfer ownership of the underlying fixed buffers
    fn take_buffers(&mut self) -> Vec<FixedSizeStateBuffer<Self::Env>>;

    // set the undelting fixed buffers
    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>);

    // TODO: probably don't really need this in the future
    fn to_rollout_buffers(&self) -> Vec<RolloutBuffer>;
}

pub enum FixedSizeEnvPoolKind<E: Env> {
    FixedSizeVecEnvPool(FixedSizeVecEnvPool<E>),
    FixedSizeThreadEnvPool(FixedSizeThreadEnvPool<E>),
}

impl<E: Env<Tensor = Buffer>> FixedSizeEnvPool for FixedSizeEnvPoolKind<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        todo!()
    }

    fn step<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        todo!()
    }

    fn to_rollout_buffers(&self) -> Vec<RolloutBuffer> {
        todo!()
    }

    fn take_buffers(&mut self) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        todo!()
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        todo!()
    }
}

pub trait VariableSizedEnvPool {
    type Env: Env<Tensor = Buffer>;

    fn num_envs(&self) -> usize;

    // TODO: should be removed once we have a trait for the trajectory buffers
    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer>;

    fn step_with_episode_bound<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        steps: usize,
    );
}

pub enum VariableSizedEnvPoolKind<E: Env> {
    VariableSizedVecEnvPool(VariableSizedVecEnvPool<E>),
    VariableSizedThreadEnvPool(VariableSizedThreadEnvPool<E>),
}

impl<E: Env<Tensor = Buffer>> VariableSizedEnvPool for VariableSizedEnvPoolKind<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        todo!()
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer> {
        todo!()
    }

    fn step_with_episode_bound<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        steps: usize,
    ) {
        todo!()
    }
}
