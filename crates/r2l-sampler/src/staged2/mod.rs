mod normalized_pool;
mod worker;

use itertools::Itertools;
use r2l_core::{
    buffers::buffer::{TrajectoryBuffer, TrajectoryView},
    env::{
        Env, EnvBuilder, EnvBuilderType,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    models::Actor,
    on_policy::algorithm::Sampler,
};
pub use worker::WorkerPool2;

use crate::{
    RolloutMode, SamplerExecutionMode, SamplerHookResult, staged2::normalized_pool::NormalizedPool,
};

/// Hook controlling rollout collection for [`StagedSampler2`].
pub trait StagedSampler2Hook {
    /// Environment type sampled by the hook.
    type E: Env;

    /// Returns the next collection instruction.
    fn hook(&mut self, core: &mut StagedSamplerCore2<Self::E>) -> SamplerHookResult;

    /// Resets hook state before a new collection run.
    fn reset(&mut self) {}
}

/// Buffered sampler state built on top of a [`NormalizedPool`].
pub struct StagedSamplerCore2<E: Env> {
    /// Actor/environment interaction pool.
    pub pool: NormalizedPool<E>,
    /// Per-environment output trajectory buffers.
    pub buffers: Vec<TrajectoryBuffer<E::Tensor>>,
}

impl<E: Env> StagedSamplerCore2<E> {
    /// Builds normalized interaction and buffering state.
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: ClippedNormalizer<E::Tensor>,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        Self {
            pool: NormalizedPool::build(env_builder, execution_mode, obs_normalizer),
            buffers: vec![TrajectoryBuffer::default(); num_envs],
        }
    }

    /// Collects a bounded rollout.
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
                    let indices = episode_counts
                        .iter()
                        .positions(|count| *count < n_episodes)
                        .collect::<Vec<_>>();
                    if indices.is_empty() {
                        break;
                    }
                    let memories = self.pool.step_indexed(&indices);
                    for (idx, memory) in indices.into_iter().zip(memories) {
                        if memory.is_done() {
                            episode_counts[idx] += 1;
                        }
                        self.buffers[idx].push(memory);
                    }
                }
            }
        }
    }

    fn step(&mut self) {
        let memories = self.pool.step();
        for (idx, memory) in memories.into_iter().enumerate() {
            self.buffers[idx].push(memory)
        }
    }

    /// Clears all output buffers.
    pub fn clear_buffers(&mut self) {
        self.buffers.iter_mut().for_each(TrajectoryBuffer::clear);
    }

    /// Installs a clone of `policy` on every worker.
    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        self.pool.set_policy(policy);
    }

    /// Borrows collected trajectories in worker order.
    pub fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, E::Tensor>]> {
        self.buffers
            .iter()
            .map(TrajectoryBuffer::to_trajectory_view)
            .collect::<Vec<_>>()
    }

    /// Stops threaded workers.
    pub fn shutdown(&mut self) {
        self.pool.shutdown();
    }
}

/// Observation-normalizing sampler using the second-generation worker pool.
pub struct StagedSampler2<E: Env, H: StagedSampler2Hook<E = E>> {
    core: StagedSamplerCore2<E>,
    hook: H,
}

impl<E: Env, H: StagedSampler2Hook<E = E>> StagedSampler2<E, H> {
    /// Builds a sampler with a mandatory observation normalizer.
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        hook: H,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: ClippedNormalizer<E::Tensor>,
    ) -> Self {
        Self {
            core: StagedSamplerCore2::build(env_builder, execution_mode, obs_normalizer),
            hook,
        }
    }

    /// Clones the shared observation normalizer with the requested access mode.
    pub fn obs_normalizer(&self, mode: NormalizerMode) -> ClippedNormalizer<E::Tensor> {
        self.core.pool.obs_normalizer(mode)
    }
}

impl<E: Env, H: StagedSampler2Hook<E = E>> Sampler for StagedSampler2<E, H> {
    type Tensor = E::Tensor;

    fn reset_all_envs(&mut self) {
        self.core.pool.reset_all();
        self.core.clear_buffers();
        self.hook.reset();
    }

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        self.core.clear_buffers();
        self.core.set_policy(actor);
        while let SamplerHookResult::Bound(bound) = self.hook.hook(&mut self.core) {
            self.core.collect(bound);
        }
    }

    fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, Self::Tensor>]> {
        self.core.trajectory_views()
    }

    fn shutdown(&mut self) {
        self.core.shutdown();
    }
}
