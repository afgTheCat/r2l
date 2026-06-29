// Fun programming. This is very reminacsent to the the VecNormalize in sb3
//
// Idea: implement a direct way of normalizing the environment. Workers not have access to the same
// buffer, instead they return the observation and the reward directly. Normalizaiton happens
// afterwards.

mod clipped_noramlizer;
mod worker;

use bimodal_array::{ArrayHandle, bimodal_array, bimodal_array_with_factory};
use r2l_core::{
    buffers::buffer::{TrajectoryBuffer, TrajectoryView},
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::Sampler,
    rng::RNG,
    tensor::R2lTensor,
};
use rand::RngExt;

use crate::{
    SamplerExecutionMode,
    normalized::{
        clipped_noramlizer::ClippedNormalizer,
        worker::{ThreadWorkerFactory, ThreadWorkers, VecWorkers, WorkerPool},
    },
};

pub struct R2lNormalizedSampler<E: Env<Tensor: R2lTensor>> {
    pool: WorkerPool<E>,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    reward_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    last_states: ArrayHandle<E::Tensor>,
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
        _with_obs_normalizer: bool,
        _with_reward_normalizer: bool,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = vec![TrajectoryBuffer::default(); num_envs];
        let (last_states, pool) = match execution_mode {
            SamplerExecutionMode::Vec => Self::build_vec_workers(env_builder, num_envs),
            SamplerExecutionMode::Thread => Self::build_thread_workers(env_builder, num_envs),
        };
        Self {
            buffers,
            pool,
            last_states,
            obs_normalizer: None,
            reward_normalizer: None,
            n_steps,
        }
    }

    // TODO: ugly! will need to make this nicer
    fn build_vec_workers<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        num_envs: usize,
    ) -> (ArrayHandle<E::Tensor>, WorkerPool<E>) {
        let mut envs_and_states = Vec::with_capacity(num_envs);
        for env_idx in 0..num_envs {
            let mut env = env_builder.build_idx(env_idx).unwrap();
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            let state = env.reset(seed).unwrap();
            envs_and_states.push((env, state));
        }
        let initial_states = envs_and_states
            .iter()
            .map(|(_, state)| state.clone())
            .collect();
        let (last_states, last_state_handles) = bimodal_array(initial_states);
        let workers = envs_and_states
            .into_iter()
            .zip(last_state_handles)
            .map(|((env, _), handle)| (env, handle))
            .collect();
        (last_states, WorkerPool::Vec(VecWorkers::new(workers)))
    }

    fn build_thread_workers<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        num_envs: usize,
    ) -> (ArrayHandle<E::Tensor>, WorkerPool<E>) {
        let (command_tx, command_rx) = crossbeam::channel::unbounded();
        let (result_tx, result_rx) = crossbeam::channel::unbounded();
        let factories = (0..num_envs)
            .map(|idx| {
                let env_builder = env_builder.clone();
                let env_builder = move || env_builder.build_idx(idx);
                ThreadWorkerFactory::new(command_rx.clone(), result_tx.clone(), env_builder.clone())
            })
            .collect();
        let last_states = bimodal_array_with_factory(factories);
        let workers = ThreadWorkers::new(num_envs, command_tx, result_rx);
        (last_states, WorkerPool::Thread(workers))
    }

    fn step(&mut self) {
        let mut multi_memory = self.pool.step();
        multi_memory.next_states = if let Some(obs_normalizer) = self.obs_normalizer.as_mut() {
            let next_states = std::mem::take(&mut multi_memory.next_states);
            let normalized_next_states = obs_normalizer.update_and_normalize(&next_states);
            for (last_state, normalized_next_state) in self
                .last_states
                .lock()
                .unwrap()
                .iter_mut()
                .zip(normalized_next_states.iter())
            {
                *last_state = normalized_next_state.clone();
            }
            normalized_next_states
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
