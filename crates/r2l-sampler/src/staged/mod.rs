// This is similar to VecNormalize in SB3.
//
// Workers do not write directly to the output buffer; they return observations and rewards.
// Normalization happens
// afterwards.

mod worker;

use bimodal_array::{ArrayHandle, bimodal_array, bimodal_array_with_factory};
use itertools::Itertools;
use r2l_core::{
    buffers::buffer::{TrajectoryBuffer, TrajectoryView},
    env::{
        Env, EnvBuilder, EnvBuilderType,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    models::Actor,
    on_policy::algorithm::Sampler,
    rng::sample_u64,
    tensor::R2lTensor,
};

use crate::{
    RolloutMode, SamplerExecutionMode, SamplerHookResult,
    staged::{
        // clipped_normalizer::ClippedNormalizer,
        worker::ThreadHandle,
        worker::{ThreadWorkerFactory, ThreadWorkers, VecWorkers, WorkerPool},
    },
};

/// Hook that controls the sequence of collection bounds for a normalized sampler.
pub trait StagedSamplerHook {
    /// Environment type sampled by the hook's sampler.
    type E: Env<Tensor: R2lTensor>;

    /// Returns the next collection instruction.
    fn hook(&mut self, core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult;

    /// Resets hook state before a new training or evaluation run.
    fn reset(&mut self) {}
}

/// Mutable normalized-sampler state exposed to hook implementations.
pub struct StagedSamplerCore<E: Env<Tensor: R2lTensor>> {
    /// Inline or threaded environment workers.
    pub pool: WorkerPool<E>,
    /// Optional shared observation normalizer.
    pub obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    /// Most recent, normalized observation for each environment.
    pub last_states: ArrayHandle<E::Tensor>,
    /// Per-environment output trajectory buffers.
    pub buffers: Vec<TrajectoryBuffer<E::Tensor>>,
}

impl<E: Env<Tensor: R2lTensor>> StagedSamplerCore<E> {
    /// Builds normalized sampler state and its environment workers.
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = vec![TrajectoryBuffer::default(); num_envs];
        let (mut last_states, pool) = match execution_mode {
            SamplerExecutionMode::SingleThreaded => Self::build_vec_workers(env_builder, num_envs),
            SamplerExecutionMode::MultiThreaded => {
                Self::build_thread_workers(env_builder, num_envs)
            }
        };
        if let Some(obs_normalizer) = &obs_normalizer {
            let mut last_states = last_states.lock().unwrap();
            obs_normalizer.apply_in_place(&mut last_states);
        }
        Self {
            buffers,
            pool,
            last_states,
            obs_normalizer,
        }
    }

    fn build_vec_workers<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        num_envs: usize,
    ) -> (ArrayHandle<E::Tensor>, WorkerPool<E>) {
        let mut envs = Vec::with_capacity(num_envs);
        let mut initial_states = Vec::with_capacity(num_envs);
        for env_idx in 0..num_envs {
            let mut env = env_builder.build_idx(env_idx).unwrap();
            let state = env.reset(sample_u64()).unwrap();
            initial_states.push(state.clone());
            envs.push(env);
        }
        let (last_states, last_state_handles) = bimodal_array(initial_states);
        let workers = envs.into_iter().zip(last_state_handles).collect();
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
                ThreadWorkerFactory::new(command_rx, result_tx, env_builder.clone(), sample_u64())
            })
            .collect();
        let last_states = bimodal_array_with_factory(factories);
        let workers = ThreadWorkers::new(worker_handles);
        (last_states, WorkerPool::Thread(workers))
    }

    /// Collects a bounded rollout from the worker pool.
    pub fn collect(&mut self, bound: RolloutMode) {
        match bound {
            RolloutMode::StepBound { n_steps } => {
                for _ in 0..n_steps {
                    self.step();
                }
            }
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episode_counts = vec![0; self.buffers.len()];
                loop {
                    let worker_idxs = episode_counts
                        .iter()
                        .positions(|count| *count < n_episodes)
                        .collect::<Vec<_>>();
                    if worker_idxs.is_empty() {
                        break;
                    }
                    let terminations = self.step_indexed(&worker_idxs);
                    for (idx, terminated) in worker_idxs.into_iter().zip(terminations) {
                        if terminated {
                            episode_counts[idx] += 1;
                        }
                    }
                }
            }
        }
    }

    fn step_indexed(&mut self, indices: &[usize]) -> Vec<bool> {
        let multi_memory = self.pool.step_indexed(indices);
        if let Some(obs_normalizer) = &self.obs_normalizer {
            let mut last_states = self.last_states.lock().unwrap();
            let mut next_states = indices
                .iter()
                .map(|idx| last_states[*idx].clone())
                .collect::<Vec<_>>();
            obs_normalizer.apply_in_place(&mut next_states);
            for (idx, next_state) in indices.iter().zip(next_states) {
                last_states[*idx] = next_state;
            }
        }
        let last_states = self.last_states.lock().unwrap();
        let next_states = indices
            .iter()
            .map(|idx| last_states[*idx].clone())
            .collect::<Vec<_>>();
        let memories = multi_memory.into_memories(&next_states);
        let terminations = memories.iter().map(|memory| memory.is_done()).collect();
        for (idx, memory) in indices.iter().zip(memories) {
            self.buffers[*idx].push(memory)
        }
        terminations
    }

    fn step(&mut self) -> Vec<bool> {
        let multi_memory = self.pool.step();
        if let Some(obs_normalizer) = &self.obs_normalizer {
            let mut last_states = self.last_states.lock().unwrap();
            obs_normalizer.apply_in_place(&mut last_states)
        }
        let last_states = self.last_states.lock().unwrap();
        let memories = multi_memory.into_memories(&last_states);
        let terminations = memories.iter().map(|memory| memory.is_done()).collect();
        for (idx, memory) in memories.into_iter().enumerate() {
            self.buffers[idx].push(memory);
        }
        terminations
    }

    /// Clears all output trajectory buffers.
    pub fn clear_buffers(&mut self) {
        self.buffers.iter_mut().for_each(|buffer| buffer.clear());
    }

    /// Installs a clone of `policy` on every worker.
    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        self.pool.set_policy(policy);
    }

    /// Borrows all collected trajectories in worker order.
    pub fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, E::Tensor>]> {
        self.buffers
            .iter()
            .map(|buffer| buffer.to_trajectory_view())
            .collect::<Vec<_>>()
    }

    /// Stops threaded workers.
    pub fn shutdown(&mut self) {
        self.pool.shutdown();
    }
}

/// Observation-normalizing rollout sampler controlled by a hook.
pub struct StagedSampler<E: Env<Tensor: R2lTensor>, H: StagedSamplerHook<E = E>> {
    core: StagedSamplerCore<E>,
    hook: H,
}

impl<E: Env<Tensor: R2lTensor>, H: StagedSamplerHook<E = E>> StagedSampler<E, H> {
    /// Builds a sampler with an existing shared observation normalizer.
    pub fn build_with_obs_normalizer<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        hook: H,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Self {
        Self {
            core: StagedSamplerCore::build(env_builder, execution_mode, obs_normalizer),
            hook,
        }
    }

    /// Clones the shared observation normalizer with the requested access mode.
    pub fn obs_normalizer(&self, mode: NormalizerMode) -> Option<ClippedNormalizer<E::Tensor>> {
        self.core
            .obs_normalizer
            .as_ref()
            .map(|normalizer| normalizer.with_mode(mode))
    }
}

impl<E: Env<Tensor: R2lTensor>, H: StagedSamplerHook<E = E>> Sampler for StagedSampler<E, H> {
    type Tensor = E::Tensor;

    fn reset_all_envs(&mut self) {
        self.core.pool.reset_all();
        if let Some(obs_normalizer) = &self.core.obs_normalizer {
            let mut last_states = self.core.last_states.lock().unwrap();
            obs_normalizer.apply_in_place(&mut last_states);
        }
        self.core.clear_buffers();
        self.hook.reset();
    }

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        self.core.clear_buffers();
        self.core.set_policy(actor.clone());
        loop {
            let result = self.hook.hook(&mut self.core);
            match result {
                SamplerHookResult::Bound(bound) => self.core.collect(bound),
                SamplerHookResult::Stop => break,
            }
        }
    }

    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]> {
        self.core.trajectory_views()
    }

    fn shutdown(&mut self) {
        self.core.shutdown();
    }
}
