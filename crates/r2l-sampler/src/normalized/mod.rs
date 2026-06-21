// Fun programming. This is very reminacsent to the the VecNormalize in sb3
//
// Idea: implement a direct way of normalizing the environment. Workers not have access to the same
// buffer, instead they return the observation and the reward directly. Normalizaiton happens
// afterwards.

mod clipped_noramlizer;
mod worker;

use std::sync::MutexGuard;

use itertools::izip;
use r2l_core::{
    buffers::{
        Memory, MultiMemory,
        buffer::{TrajectoryBuffer, TrajectoryView},
    },
    env::Env,
    models::Actor,
    on_policy::algorithm::Sampler,
    tensor::RunningMeanTensor,
};

use crate::normalized::{clipped_noramlizer::ClippedNormalizer, worker::WorkerPool};

const EPS: f32 = 1e-8;

struct Coordinator<E: Env<Tensor: RunningMeanTensor>> {
    pool: WorkerPool<E>,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    reward_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    // Here there is no need to have each thread own the buffer
    buffers: Vec<TrajectoryBuffer<E::Tensor>>,
}

impl<E: Env<Tensor: RunningMeanTensor>> Coordinator<E> {
    fn step(&mut self) {
        let mut multi_memory = self.pool.step();
        multi_memory.next_states = if let Some(obs_normalizer) = self.obs_normalizer.as_mut() {
            // TODO: update should be able to have a vector passed in. Will have to check what this
            // means in reality.
            // obs_normalizer.rm.update(&obs);
            // TODO: In sb3, this is normalized either way
            obs_normalizer.normalize(std::mem::take(&mut multi_memory.next_states))
        } else {
            std::mem::take(&mut multi_memory.next_states)
        };
        let memories = multi_memory.into_memories();
        for (idx, memory) in memories.into_iter().enumerate() {
            self.buffers[idx].push(memory);
        }
    }
}

impl<E: Env<Tensor: RunningMeanTensor>> Sampler for Coordinator<E> {
    type Tensor = E::Tensor;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        todo!()
    }

    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]> {
        self.buffers
            .iter()
            .map(|b| b.to_trajectory_view())
            .collect::<Vec<_>>()
    }

    fn shutdown(&mut self) {
        todo!()
    }
}
