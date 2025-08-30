pub mod env_pools;
pub mod trajectory_buffers;

use crate::{
    distributions::Distribution,
    env::{Env, EnvironmentDescription, Sampler},
    numeric::Buffer,
    sampler::{
        env_pools::{
            FixedSizeEnvPool, FixedSizeEnvPoolKind, VariableSizedEnvPool, VariableSizedEnvPoolKind,
        },
        trajectory_buffers::fixed_size_buffer::FixedSizeStateBuffer,
    },
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Result, Tensor};
use std::{fmt::Debug, marker::PhantomData};

// TODO: this is not a bad idea. However in the future we do not want a reference here, but an
// Arc::RwLock for the underlying distribution.
struct DistributionWrapper<'a, D: Distribution, E: Env> {
    distribution: &'a D,
    env: PhantomData<E>,
}

impl<'a, D: Distribution, E: Env> DistributionWrapper<'a, D, E> {
    fn new(distribution: &'a D) -> Self {
        Self {
            distribution,
            env: PhantomData,
        }
    }
}

// SAFETY: This can be safely shared between threads as the env is just a PhantomData
unsafe impl<'a, D: Distribution, E: Env> Sync for DistributionWrapper<'a, D, E> {}

impl<'a, D: Distribution, E: Env> Debug for DistributionWrapper<'a, D, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<'a, D: Distribution, E: Env> Distribution for DistributionWrapper<'a, D, E>
where
    E::Tensor: From<D::Tensor>,
    E::Tensor: Into<D::Tensor>,
{
    type Tensor = E::Tensor;

    fn std(&self) -> Result<f32> {
        self.distribution.std()
    }

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.distribution.get_action(observation.into())?;
        Ok(action.into())
    }

    fn log_probs(&self, states: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        let log_probs = self.distribution.log_probs(states.into(), actions.into())?;
        Ok(log_probs.into())
    }

    fn entropy(&self) -> Result<Self::Tensor> {
        let entropy = self.distribution.entropy()?;
        Ok(entropy.into())
    }

    fn resample_noise(&mut self) -> candle_core::Result<()> {
        // TODO: we may want the distribution to be behind a RwLock, but I doubt that this will be
        // called a whole lot. In future releases we should enable finer control of noise sampling
        todo!()
    }
}

pub trait SequntialStepBoundHooks<E: Env> {
    fn process_last_step(
        &mut self,
        distr: &dyn Distribution<Tensor = E::Tensor>,
        buffers: &mut Vec<FixedSizeStateBuffer<E>>,
    );

    // TODO: this should change
    fn post_process_hook(&self) {}
}

pub enum CollectionType<E: Env> {
    StepBound {
        env_pool: FixedSizeEnvPoolKind<E>,
        hooks: Option<Box<dyn SequntialStepBoundHooks<E>>>,
    },
    EpisodeBound {
        env_pool: VariableSizedEnvPoolKind<E>,
    },
}

// what I can live with two different different structs here
pub struct NewSampler<E: Env> {
    pub env_steps: usize,
    pub collection_type: CollectionType<E>,
}

impl<E: Env> NewSampler<E> {
    pub fn env_description(&self) -> EnvironmentDescription {
        match &self.collection_type {
            CollectionType::StepBound { env_pool, .. } => env_pool.env_description(),
            CollectionType::EpisodeBound { env_pool } => env_pool.env_description(),
        }
    }
}

impl<E: Env<Tensor = Buffer>> Sampler for NewSampler<E> {
    type Env = E;

    fn collect_rollouts<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Result<Vec<RolloutBuffer<Tensor>>> {
        let distr: DistributionWrapper<D, E> = DistributionWrapper::new(distr);
        match &mut self.collection_type {
            CollectionType::StepBound { env_pool, hooks } => {
                if let Some(hooks) = hooks {
                    let mut steps_taken = 0;
                    while steps_taken < self.env_steps {
                        let mut buffers = env_pool.step_take_buffers(&distr);
                        hooks.process_last_step(&distr, &mut buffers);
                        env_pool.set_buffers(buffers);
                        steps_taken += 1;
                    }
                } else {
                    env_pool.step(&distr, self.env_steps);
                }
                Ok(env_pool
                    .to_rollout_buffers()
                    .into_iter()
                    .map(|rb| rb.convert())
                    .collect())
            }
            CollectionType::EpisodeBound { env_pool } => {
                env_pool.step_with_episode_bound(&distr, self.env_steps);
                Ok(env_pool.to_rollout_buffers())
            }
        }
    }
}
