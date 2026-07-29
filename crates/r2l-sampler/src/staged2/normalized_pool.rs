use r2l_core::{
    buffers::Memory,
    env::{
        Env, EnvBuilder, EnvBuilderType,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    models::Actor,
};

use crate::{SamplerExecutionMode, staged2::WorkerPool2};

/// Raw worker pool decorated with mandatory observation normalization.
pub struct NormalizedPool<E: Env> {
    pool: WorkerPool2<E>,
    obs_normalizer: ClippedNormalizer<E::Tensor>,
}

impl<E: Env> NormalizedPool<E> {
    /// Builds a worker pool and normalizes its initial observations.
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: ClippedNormalizer<E::Tensor>,
    ) -> Self {
        let mut pool = WorkerPool2::build(env_builder, execution_mode);
        obs_normalizer.apply_slice_in_place(&mut pool.last_state_mut());
        Self {
            pool,
            obs_normalizer,
        }
    }

    /// Steps every worker and returns transitions with normalized observations.
    pub fn step(&mut self) -> Vec<Memory<E::Tensor>> {
        let mut memories = self.pool.step();
        let mut states = self.pool.last_state_mut();
        self.obs_normalizer.apply_slice_in_place(&mut states);
        for (memory, state) in memories.iter_mut().zip(states.iter()) {
            memory.next_state = state.clone()
        }
        memories
    }

    /// Steps selected workers and returns transitions in `indices` order.
    pub fn step_indexed(&mut self, indices: &[usize]) -> Vec<Memory<E::Tensor>> {
        if indices.is_empty() {
            return Vec::new();
        }
        let mut memories = self.pool.step_indexed(indices);
        let mut states = self.pool.last_state_mut();
        let mut next_states = indices
            .iter()
            .map(|idx| states[*idx].clone())
            .collect::<Vec<_>>();
        self.obs_normalizer.apply_slice_in_place(&mut next_states);
        for ((idx, memory), next_state) in indices.iter().zip(&mut memories).zip(next_states) {
            states[*idx] = next_state.clone();
            memory.next_state = next_state;
        }
        memories
    }

    /// Installs a clone of `policy` on every worker.
    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        self.pool.set_policy(policy);
    }

    /// Resets and normalizes every environment.
    pub fn reset_all(&mut self) {
        self.pool.reset_all();
        let mut states = self.pool.last_state_mut();
        self.obs_normalizer.apply_slice_in_place(&mut states)
    }

    /// Resets and normalizes selected environments.
    pub fn reset_indexed(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }
        self.pool.reset_indexed(indices);
        let mut states = self.pool.last_state_mut();
        let mut reset_states = indices
            .iter()
            .map(|idx| states[*idx].clone())
            .collect::<Vec<_>>();
        self.obs_normalizer.apply_slice_in_place(&mut reset_states);
        for (idx, state) in indices.iter().zip(reset_states) {
            states[*idx] = state
        }
    }

    /// Clones current normalized observations in worker order.
    pub fn current_states(&mut self) -> Vec<E::Tensor> {
        self.pool.current_states()
    }

    /// Returns the number of environments.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Returns whether the pool contains no environments.
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Stops threaded workers.
    pub fn shutdown(&mut self) {
        self.pool.shutdown();
    }
}
