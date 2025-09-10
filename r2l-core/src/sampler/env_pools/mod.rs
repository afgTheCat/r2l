pub mod thread_env_pool;
pub mod vec_env_pool;

use crate::{
    distributions::Policy,
    env::{Env, EnvironmentDescription},
    sampler::{
        env_pools::{
            thread_env_pool::{FixedSizeThreadEnvPool, VariableSizedThreadEnvPool},
            vec_env_pool::{FixedSizeVecEnvPool, VariableSizedVecEnvPool},
        },
        trajectory_buffers::fixed_size_buffer::FixedSizeStateBuffer,
    },
    utils::rollout_buffer::RolloutBuffer,
};

pub trait FixedSizeEnvPool {
    type Env: Env;

    // num envs
    fn num_envs(&self) -> usize;

    /// async mode
    fn step_n<D: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>>;

    /// step mode
    fn step_take_buffers(&mut self) -> Vec<FixedSizeStateBuffer<Self::Env>>;

    /// set the buffers
    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>);

    /// set the distribution
    fn set_distr<D: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(&mut self, distr: D);

    // take rollout buffer
    fn take_rollout_buffers(&mut self) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>>;
}

pub enum FixedSizeEnvPoolKind<E: Env> {
    FixedSizeVecEnvPool(FixedSizeVecEnvPool<E>),
    FixedSizeThreadEnvPool(FixedSizeThreadEnvPool<E>),
}

impl<E: Env> FixedSizeEnvPoolKind<E> {
    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        match self {
            Self::FixedSizeVecEnvPool(env_pool) => env_pool.env_description(),
            Self::FixedSizeThreadEnvPool(env_pool) => env_pool.env_description(),
        }
    }
}

impl<E: Env> FixedSizeEnvPool for FixedSizeEnvPoolKind<E> {
    type Env = E;

    fn step_n<D: Policy<Tensor = E::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<E::Tensor>> {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.step_n(distr, steps),
            Self::FixedSizeThreadEnvPool(pool) => pool.step_n(distr, steps),
        }
    }

    fn num_envs(&self) -> usize {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.num_envs(),
            Self::FixedSizeThreadEnvPool(pool) => pool.num_envs(),
        }
    }

    fn step_take_buffers(&mut self) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.step_take_buffers(),
            Self::FixedSizeThreadEnvPool(pool) => pool.step_take_buffers(),
        }
    }

    fn set_distr<D: Policy<Tensor = E::Tensor> + Clone>(&mut self, distr: D) {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.set_distr(distr),
            Self::FixedSizeThreadEnvPool(pool) => pool.set_distr(distr),
        }
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.set_buffers(buffers),
            Self::FixedSizeThreadEnvPool(pool) => pool.set_buffers(buffers),
        }
    }

    fn take_rollout_buffers(&mut self) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.take_rollout_buffers(),
            Self::FixedSizeThreadEnvPool(pool) => pool.take_rollout_buffers(),
        }
    }
}

pub trait VariableSizedEnvPool {
    type Env: Env;

    fn num_envs(&self) -> usize;

    fn step_with_episode_bound<D: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>>;
}

pub enum VariableSizedEnvPoolKind<E: Env> {
    VariableSizedVecEnvPool(VariableSizedVecEnvPool<E>),
    VariableSizedThreadEnvPool(VariableSizedThreadEnvPool<E>),
}

impl<E: Env> VariableSizedEnvPoolKind<E> {
    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        match self {
            Self::VariableSizedVecEnvPool(env_pool) => env_pool.buffers[0].env.env_description(),
            Self::VariableSizedThreadEnvPool(env_pool) => env_pool.env_description(),
        }
    }
}

impl<E: Env> VariableSizedEnvPool for VariableSizedEnvPoolKind<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        match self {
            Self::VariableSizedVecEnvPool(pool) => pool.num_envs(),
            Self::VariableSizedThreadEnvPool(pool) => pool.num_envs(),
        }
    }

    fn step_with_episode_bound<D: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        match self {
            Self::VariableSizedVecEnvPool(pool) => pool.step_with_episode_bound(distr, steps),
            Self::VariableSizedThreadEnvPool(pool) => pool.step_with_episode_bound(distr, steps),
        }
    }
}
