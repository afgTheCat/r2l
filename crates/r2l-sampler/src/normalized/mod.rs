// Fun programming. This is very reminacsent to the the VecNormalize in sb3
//
// Idea: implement a direct way of normalizing the environment. Workers not have access to the same
// buffer, instead they return the observation and the reward directly. Normalizaiton happens
// afterwards.

mod clipped_noramlizer;
mod worker;

use std::sync::Arc;

use r2l_core::{
    buffers::buffer::{TrajectoryBuffer, TrajectoryView},
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::Sampler,
    tensor::R2lTensor,
};

use crate::{
    SamplerExecutionMode,
    normalized::{
        clipped_noramlizer::ClippedNormalizer,
        worker::{ThreadHandle, ThreadWorker, ThreadWorkers, VecWorkers, Worker, WorkerPool},
    },
};

pub struct R2lNormalizedSampler<E: Env<Tensor: R2lTensor>> {
    pool: WorkerPool<E>,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    reward_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    // Here there is no need to have each thread own the buffer
    buffers: Vec<TrajectoryBuffer<E::Tensor>>,
    // TODO: we might want later on. Maybe other things?
    n_steps: usize,
}

impl<E: Env<Tensor: R2lTensor>> R2lNormalizedSampler<E> {
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        n_steps: usize,
        execution_mode: SamplerExecutionMode,
        with_obs_normalizer: bool,
        with_reward_normalizer: bool,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = vec![TrajectoryBuffer::default(); num_envs];
        let pool = match execution_mode {
            SamplerExecutionMode::Vec => {
                let vec_workers = VecWorkers::from_env_builder(env_builder);
                WorkerPool::VecCoord(vec_workers)
            }
            SamplerExecutionMode::Thread => {
                let (command_tx, command_rx) = crossbeam::channel::unbounded();
                let (res_tx, res_rx) = crossbeam::channel::unbounded();
                let num_envs = env_builder.num_envs();
                let env_builder = Arc::new(env_builder);
                let thread_handles = (0..num_envs)
                    .map(|env_idx| {
                        let env_builder = env_builder.clone();
                        let res_tx = res_tx.clone();
                        let command_rx = command_rx.clone();
                        let handle = std::thread::spawn(move || {
                            let env = env_builder.build_idx(env_idx).unwrap();
                            let worker = Worker::from_env(env);
                            let mut thread_worker = ThreadWorker::new(worker, command_rx, res_tx);
                            thread_worker.work();
                        });
                        ThreadHandle::new(handle, command_tx.clone(), res_rx.clone())
                    })
                    .collect();
                let thread_workers = ThreadWorkers::new(thread_handles);
                WorkerPool::Thread(thread_workers)
            }
        };
        Self {
            buffers,
            pool,
            obs_normalizer: None,
            reward_normalizer: None,
            n_steps,
        }
    }

    fn step(&mut self) {
        let mut multi_memory = self.pool.step();
        multi_memory.next_states = if let Some(obs_normalizer) = self.obs_normalizer.as_mut() {
            let next_states = std::mem::take(&mut multi_memory.next_states);
            obs_normalizer.update_and_normalize(&next_states)
        } else {
            std::mem::take(&mut multi_memory.next_states)
        };

        // TODO: add this once the normalizer is working as intended
        // multi_memory.rewards = if let Some(rew_normalizer) = self.reward_normalizer.as_mut() {
        //     rew_normalizer.normalize(std::mem::take(&mut multi_memory.rewards))
        // } else {
        //     std::mem::take(&mut multi_memory.rewards)
        // };
        let memories = multi_memory.into_memories();
        for (idx, memory) in memories.into_iter().enumerate() {
            self.buffers[idx].push(memory);
        }
    }
}

impl<E: Env<Tensor: R2lTensor>> Sampler for R2lNormalizedSampler<E> {
    type Tensor = E::Tensor;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        self.buffers.iter_mut().for_each(|b| b.clear());
        self.pool.set_policy(actor.clone());
        let mut steps = 0;
        while steps < self.n_steps {
            self.step();
            steps += 1;
        }
    }

    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]> {
        self.buffers
            .iter()
            .map(|b| b.to_trajectory_view())
            .collect::<Vec<_>>()
    }

    fn shutdown(&mut self) {}
}
