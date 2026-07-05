// Fun programming. This is very reminacsent to the the VecNormalize in sb3
//
// Idea: implement a direct way of normalizing the environment. Workers not have access to the same
// buffer, instead they return the observation and the reward directly. Normalizaiton happens
// afterwards.

pub mod clipped_normalizer;
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
        clipped_normalizer::ClippedNormalizer,
        worker::ThreadHandle,
        worker::{ThreadWorkerFactory, ThreadWorkers, VecWorkers, WorkerPool},
    },
};

/// Controls whether a normalized sampler mutates shared normalization stats.
#[derive(Debug, Clone, Copy)]
pub enum NormalizerMode {
    Update,
    ReadOnly,
}

pub struct R2lNormalizedSampler<E: Env<Tensor: R2lTensor>> {
    pool: WorkerPool<E>,
    pub obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
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
        with_obs_normalizer: Option<f32>,
        obs_normalizer_mode: NormalizerMode,
        _with_reward_normalizer: bool,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = vec![TrajectoryBuffer::default(); num_envs];
        let env_description = env_builder.env_description().unwrap();
        let (mut last_states, pool) = match execution_mode {
            SamplerExecutionMode::Vec => Self::build_vec_workers(env_builder, num_envs),
            SamplerExecutionMode::Thread => Self::build_thread_workers(env_builder, num_envs),
        };
        let obs_normalizer = with_obs_normalizer.map(|clip| {
            let obs_size = env_description.observation_space.size();
            let obs_normalizer = ClippedNormalizer::new(obs_normalizer_mode, clip, vec![obs_size]);
            let mut last_states = last_states.lock().unwrap();
            obs_normalizer.apply_in_place(&mut last_states);
            obs_normalizer
        });
        Self {
            buffers,
            pool,
            last_states,
            obs_normalizer,
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
        let mut worker_handles = Vec::with_capacity(num_envs);
        let factories = (0..num_envs)
            .map(|idx| {
                let (command_tx, command_rx) = crossbeam::channel::unbounded();
                let (result_tx, result_rx) = crossbeam::channel::unbounded();
                worker_handles.push(ThreadHandle::new(command_tx, result_rx));
                let env_builder = env_builder.clone();
                let env_builder = move || env_builder.build_idx(idx);
                ThreadWorkerFactory::new(command_rx, result_tx, env_builder.clone())
            })
            .collect();
        let last_states = bimodal_array_with_factory(factories);
        let workers = ThreadWorkers::new(worker_handles);
        (last_states, WorkerPool::Thread(workers))
    }

    fn step(&mut self) {
        let multi_memory = self.pool.step();
        if let Some(obs_normalizer) = &self.obs_normalizer {
            let mut last_states = self.last_states.lock().unwrap();
            obs_normalizer.apply_in_place(&mut last_states)
        }

        // TODO: add this once the normalizer is working as intended
        // multi_memory.rewards = if let Some(rew_normalizer) = self.reward_normalizer.as_mut() {
        //     rew_normalizer.normalize(std::mem::take(&mut multi_memory.rewards))
        // } else {
        //     std::mem::take(&mut multi_memory.rewards)
        // };
        let last_states = self.last_states.lock().unwrap();
        let memories = multi_memory.into_memories(&last_states);
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

    fn shutdown(&mut self) {
        self.pool.shutdown();
    }
}
