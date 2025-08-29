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

pub trait SequntialStepBoundHooks<E: Env> {
    fn process_last_step(
        &mut self,
        distr: &dyn Distribution<Tensor = Tensor>,
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
        match &mut self.collection_type {
            CollectionType::StepBound { env_pool, hooks } => {
                if let Some(hooks) = hooks {
                    let mut steps_taken = 0;
                    while steps_taken < self.env_steps {
                        let mut buffers = env_pool.step_take_buffers(distr);
                        hooks.process_last_step(distr, &mut buffers);
                        env_pool.set_buffers(buffers);
                        steps_taken += 1;
                    }
                } else {
                    env_pool.step(distr, self.env_steps);
                }
                Ok(env_pool.to_rollout_buffers())
            }
            CollectionType::EpisodeBound { env_pool } => {
                env_pool.step_with_episode_bound(distr, self.env_steps);
                Ok(env_pool.to_rollout_buffers())
            }
        }
    }
}
