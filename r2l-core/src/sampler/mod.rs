pub mod env_pools;
pub mod trajectory_buffers;

use crate::{
    distributions::Distribution,
    env::{Env, EnvironmentDescription, Sampler},
    sampler::{
        env_pools::{
            FixedSizeEnvPool, FixedSizeEnvPoolKind, VariableSizedEnvPool, VariableSizedEnvPoolKind,
        },
        trajectory_buffers::fixed_size_buffer::FixedSizeStateBuffer,
    },
    utils::rollout_buffer::RolloutBuffer,
};
use anyhow::Result;
use std::{fmt::Debug, marker::PhantomData};

// TODO: this is not a bad idea. However in the future we do not want a reference here, but an
// Arc::RwLock for the underlying distribution.
#[derive(Debug, Clone)]
pub struct DistributionWrapper<D: Distribution + Clone, T: Clone + Send + Sync + Debug + 'static> {
    distribution: D,
    env: PhantomData<T>,
}

impl<D: Distribution + Clone, T: Clone + Send + Sync + Debug + 'static> DistributionWrapper<D, T> {
    pub fn new(distribution: D) -> Self {
        Self {
            distribution,
            env: PhantomData,
        }
    }
}

// SAFETY: This can be safely shared between threads as the env is just a PhantomData
unsafe impl<D: Distribution + Clone, T: Send + Clone + Sync + Debug + 'static> Sync
    for DistributionWrapper<D, T>
{
}
// unsafe impl<D: Distribution, T: Clone + Sync + Debug + 'static> Sync for DistributionWrapper<D, T> {}

// impl<D: Distribution, E: Env> Debug for DistributionWrapper<D, E> {
//     fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         todo!()
//     }
// }

impl<D: Distribution + Clone, T: Send + Clone + Sync + Debug + 'static> Distribution
    for DistributionWrapper<D, T>
where
    T: From<D::Tensor>,
    T: Into<D::Tensor>,
{
    type Tensor = T;

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

    fn resample_noise(&mut self) -> Result<()> {
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

pub struct NewSampler<E: Env> {
    pub env_steps: usize,
    pub collection_type: CollectionType<E>,
}

impl<E: Env> NewSampler<E> {
    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        match &self.collection_type {
            CollectionType::StepBound { env_pool, .. } => env_pool.env_description(),
            CollectionType::EpisodeBound { env_pool } => env_pool.env_description(),
        }
    }
}

impl<E: Env> Sampler for NewSampler<E> {
    type Env = E;

    fn collect_rollouts<D: Distribution + Clone>(
        &mut self,
        distr: D,
    ) -> Result<Vec<RolloutBuffer<D::Tensor>>>
    where
        E::Tensor: From<D::Tensor>,
        E::Tensor: Into<D::Tensor>,
    {
        let distr: DistributionWrapper<D, E::Tensor> = DistributionWrapper::new(distr);
        let rb = match &mut self.collection_type {
            CollectionType::StepBound { env_pool, hooks } => {
                if let Some(hooks) = hooks {
                    env_pool.set_distr(distr.clone());
                    let mut steps_taken = 0;
                    while steps_taken < self.env_steps {
                        let mut buffers = env_pool.step_take_buffers();
                        hooks.process_last_step(&distr, &mut buffers);
                        env_pool.set_buffers(buffers);
                        steps_taken += 1;
                    }
                    env_pool.take_rollout_buffers()
                } else {
                    env_pool.step_n(distr, self.env_steps)
                }
            }
            CollectionType::EpisodeBound { env_pool } => {
                env_pool.step_with_episode_bound(distr, self.env_steps)
            }
        };
        Ok(rb.into_iter().map(|rb| rb.convert()).collect())
    }
}
