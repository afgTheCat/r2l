pub mod step_bound_sampler;

use crate::{
    distributions::Distribution,
    env::{Env, Sampler},
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

enum CollectionType<E: Env> {
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
    env_steps: usize,
    collection_type: CollectionType<E>,
}

impl<E: Env<Tensor = Buffer>> Sampler for NewSampler<E> {
    fn collect_rollouts<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Result<Vec<RolloutBuffer>> {
        match &mut self.collection_type {
            CollectionType::StepBound { env_pool, hooks } => {
                if let Some(hooks) = hooks {
                    let mut steps_taken = 0;
                    while steps_taken < self.env_steps {
                        env_pool.step(distr, 1);
                        let mut buffers = env_pool.take_buffers();
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
