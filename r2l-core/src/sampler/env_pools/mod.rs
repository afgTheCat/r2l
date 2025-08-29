pub mod thread_env_pool;
pub mod vec_env_pool;

use crate::{
    distributions::Distribution,
    env::{Env, EnvironmentDescription},
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
use std::{fmt::Debug, marker::PhantomData};

// wraps a distribution and auto converts
struct DistributionWrapper<'a, D: Distribution, E: Env> {
    distribution: &'a D,
    env: PhantomData<E>,
}

// SAFETY: This can be safely shared between threads as the env is just a PhantomData
unsafe impl<'a, D: Distribution, E: Env> Sync for DistributionWrapper<'a, D, E> {}

impl<'a, D: Distribution, E: Env> Debug for DistributionWrapper<'a, D, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

// impl<'a, D: Distribution, E: Env> Distribution for DistributionWrapper<'a, D, E>
// where
//     E::Tensor: From<D::Tensor>,
//     E::Tensor: Into<D::Tensor>,
// {
//     type Tensor = E::Tensor;
//
//     fn std(&self) -> Result<f32> {
//         todo!()
//     }
//
//     fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn log_probs(&self, states: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn entropy(&self) -> Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn resample_noise(&mut self) -> candle_core::Result<()> {
//         todo!()
//     }
// }

pub trait FixedSizeEnvPool {
    type Env: Env<Tensor = Buffer>;

    /// Return the number of environments
    fn num_envs(&self) -> usize;

    /// Take `steps` amount of steps using `distr`
    fn step<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize);

    /// Take one step and transfer `FixedSizeStateBuffer`
    fn step_take_buffers<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>>;

    /// Set the undelying buffer
    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>);

    // TODO: probably don't really need this in the future
    fn to_rollout_buffers<D: Clone>(&mut self) -> Vec<RolloutBuffer<D>>
    where
        <Self::Env as Env>::Tensor: From<D>,
        <Self::Env as Env>::Tensor: Into<D>;
}

pub enum FixedSizeEnvPoolKind<E: Env> {
    FixedSizeVecEnvPool(FixedSizeVecEnvPool<E>),
    FixedSizeThreadEnvPool(FixedSizeThreadEnvPool<E>),
}

impl<E: Env> FixedSizeEnvPoolKind<E> {
    pub fn env_description(&self) -> EnvironmentDescription {
        match self {
            Self::FixedSizeVecEnvPool(env_pool) => env_pool.env_description(),
            Self::FixedSizeThreadEnvPool(env_pool) => env_pool.env_description(),
        }
    }
}

impl<E: Env<Tensor = Buffer>> FixedSizeEnvPool for FixedSizeEnvPoolKind<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.num_envs(),
            Self::FixedSizeThreadEnvPool(pool) => pool.num_envs(),
        }
    }

    fn step<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.step(distr, steps),
            Self::FixedSizeThreadEnvPool(pool) => pool.step(distr, steps),
        }
    }

    fn to_rollout_buffers<D: Clone>(&mut self) -> Vec<RolloutBuffer<D>>
    where
        <Self::Env as Env>::Tensor: From<D>,
        <Self::Env as Env>::Tensor: Into<D>,
    {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.to_rollout_buffers(),
            Self::FixedSizeThreadEnvPool(pool) => pool.to_rollout_buffers(),
        }
    }

    fn step_take_buffers<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.step_take_buffers(distr),
            Self::FixedSizeThreadEnvPool(pool) => pool.step_take_buffers(distr),
        }
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        match self {
            Self::FixedSizeVecEnvPool(pool) => pool.set_buffers(buffers),
            Self::FixedSizeThreadEnvPool(pool) => pool.set_buffers(buffers),
        }
    }
}

pub trait VariableSizedEnvPool {
    type Env: Env<Tensor = Buffer>;

    fn num_envs(&self) -> usize;

    // TODO: should be removed once we have a trait for the trajectory buffers
    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer<Tensor>>;

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

impl<E: Env> VariableSizedEnvPoolKind<E> {
    pub fn env_description(&self) -> EnvironmentDescription {
        match self {
            Self::VariableSizedVecEnvPool(env_pool) => env_pool.buffers[0].env.env_description(),
            Self::VariableSizedThreadEnvPool(env_pool) => env_pool.env_description(),
        }
    }
}

impl<E: Env<Tensor = Buffer>> VariableSizedEnvPool for VariableSizedEnvPoolKind<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        match self {
            Self::VariableSizedVecEnvPool(pool) => pool.num_envs(),
            Self::VariableSizedThreadEnvPool(pool) => pool.num_envs(),
        }
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer<Tensor>> {
        match self {
            Self::VariableSizedVecEnvPool(pool) => pool.to_rollout_buffers(),
            Self::VariableSizedThreadEnvPool(pool) => pool.to_rollout_buffers(),
        }
    }

    fn step_with_episode_bound<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        steps: usize,
    ) {
        match self {
            Self::VariableSizedVecEnvPool(pool) => pool.step_with_episode_bound(distr, steps),
            Self::VariableSizedThreadEnvPool(pool) => pool.step_with_episode_bound(distr, steps),
        }
    }
}
