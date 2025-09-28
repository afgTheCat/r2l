pub mod env_pools;
pub mod trajectory_buffers;

use crate::{
    distributions::Policy,
    env::{Env, EnvironmentDescription, Sampler},
    sampler::{
        env_pools::{
            FixedSizeEnvPool, FixedSizeEnvPoolKind, VariableSizedEnvPool, VariableSizedEnvPoolKind,
        },
        trajectory_buffers::fixed_size_buffer::FixedSizeStateBuffer,
    },
    tensor::R2lTensor,
    utils::rollout_buffer::RolloutBuffer,
};
use anyhow::Result;
use std::{fmt::Debug, marker::PhantomData};

// TODO: this is not a bad idea. However in the future we do not want a reference here, but an
// Arc::RwLock for the underlying distribution.
#[derive(Debug, Clone)]
pub struct PolicyWrapper<P: Policy + Clone, T: R2lTensor> {
    policy: P,
    env: PhantomData<T>,
}

impl<D: Policy + Clone, T: R2lTensor> PolicyWrapper<D, T> {
    pub fn new(policy: D) -> Self {
        Self {
            policy,
            env: PhantomData,
        }
    }
}

impl<D: Policy + Clone, T: R2lTensor> Policy for PolicyWrapper<D, T>
where
    T: From<D::Tensor>,
    T: Into<D::Tensor>,
{
    type Tensor = T;

    fn std(&self) -> Result<f32> {
        self.policy.std()
    }

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.policy.get_action(observation.into())?;
        Ok(action.into())
    }

    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor> {
        let observations = observations
            .into_iter()
            .map(|o| o.clone().into())
            .collect::<Vec<_>>();
        let actions = actions
            .into_iter()
            .map(|a| a.clone().into())
            .collect::<Vec<_>>();
        let log_probs = self.policy.log_probs(&observations, &actions)?;
        Ok(log_probs.into())
    }

    fn entropy(&self) -> Result<Self::Tensor> {
        let entropy = self.policy.entropy()?;
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
        distr: &dyn Policy<Tensor = E::Tensor>,
        buffers: &mut Vec<FixedSizeStateBuffer<E>>,
    );

    // TODO: this should change
    fn post_process_hook(&self) {}
}

pub enum CollectionType<E: Env> {
    StepBound {
        env_pool: FixedSizeEnvPoolKind<E>,
        // TODO: this should be the preprocessor
        hooks: Option<Box<dyn SequntialStepBoundHooks<E>>>,
    },
    EpisodeBound {
        env_pool: VariableSizedEnvPoolKind<E>,
    },
}

pub struct R2lSampler<E: Env> {
    pub env_steps: usize,
    pub collection_type: CollectionType<E>,
}

impl<E: Env> R2lSampler<E> {
    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        match &self.collection_type {
            CollectionType::StepBound { env_pool, .. } => env_pool.env_description(),
            CollectionType::EpisodeBound { env_pool } => env_pool.env_description(),
        }
    }
}

impl<E: Env> Sampler for R2lSampler<E> {
    type Env = E;

    fn collect_rollouts<P: Policy + Clone>(
        &mut self,
        policy: P,
    ) -> Result<Vec<RolloutBuffer<P::Tensor>>>
    where
        E::Tensor: From<P::Tensor>,
        E::Tensor: Into<P::Tensor>,
    {
        let policy: PolicyWrapper<P, E::Tensor> = PolicyWrapper::new(policy);
        let rb = match &mut self.collection_type {
            CollectionType::StepBound { env_pool, hooks } => {
                if let Some(hooks) = hooks {
                    env_pool.set_distr(policy.clone());
                    let mut steps_taken = 0;
                    while steps_taken < self.env_steps {
                        let mut buffers = env_pool.step_take_buffers();
                        hooks.process_last_step(&policy, &mut buffers);
                        env_pool.set_buffers(buffers);
                        steps_taken += 1;
                    }
                    env_pool.take_rollout_buffers()
                } else {
                    env_pool.step_n(policy, self.env_steps)
                }
            }
            CollectionType::EpisodeBound { env_pool } => {
                env_pool.step_with_episode_bound(policy, self.env_steps)
            }
        };
        Ok(rb.into_iter().map(|rb| rb.convert()).collect())
    }
}
