// R2l sampler where each worker writes directly to the output buffer. This is preferred, when the
// raw observations and rewards are to be stored.

pub mod worker;

use std::sync::Arc;

use bimodal_array::ArrayHandle;
use bimodal_array::bimodal_array;
use r2l_core::buffers::buffer::TrajectoryBuffer;
use r2l_core::buffers::buffer::TrajectoryView;
use r2l_core::env::Env;
use r2l_core::env::EnvBuilder;
use r2l_core::env::EnvBuilderType;
use r2l_core::error::Result;
use r2l_core::models::Actor;
use r2l_core::on_policy::algorithm::Sampler;
use r2l_core::rng::{sample_u64, set_seed};

use crate::RolloutMode;
use crate::SamplerExecutionMode;
use crate::direct::worker::ThreadHandle;
use crate::direct::worker::ThreadWorker;
use crate::direct::worker::ThreadWorkers;
use crate::direct::worker::Worker;
use crate::direct::worker::WorkerPool;

/// Instruction returned by a [`DirectSamplerHook`] during rollout collection.
pub enum SamplerHookResult {
    /// Finish the current rollout.
    Stop,
    /// Collect data up to the supplied bound, then invoke the hook again.
    Bound(RolloutMode),
}

/// Hook that controls the sequence of collection bounds for a raw sampler.
pub trait DirectSamplerHook {
    /// Environment type sampled by the hook's sampler.
    type E: Env;

    /// Returns the next collection instruction.
    fn hook(&mut self, core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult;

    /// Resets hook state before a new training or evaluation run.
    fn reset(&mut self) {}
}

/// Mutable direct-sampler state exposed to [`DirectSamplerHook`] implementations.
pub struct DirectSamplerCore<E: Env> {
    buffers: ArrayHandle<TrajectoryBuffer<E::Tensor>>,
    worker_pool: WorkerPool<E>,
}

impl<E: Env> DirectSamplerCore<E> {
    /// Returns the per-environment output buffers.
    pub fn buffers_mut(&mut self) -> &mut ArrayHandle<TrajectoryBuffer<E::Tensor>> {
        &mut self.buffers
    }

    /// Resets every worker environment and clears its active episode state.
    ///
    /// # Errors
    ///
    /// Returns an error if an environment cannot be reset or a worker is interrupted.
    pub fn reset_all_envs(&mut self) -> Result<()> {
        self.worker_pool.reset_all_envs()
    }

    /// Builds sampler state from an environment collection and execution mode.
    ///
    /// # Panics
    ///
    /// Panics if an environment cannot be built.
    #[must_use]
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers: Vec<TrajectoryBuffer<E::Tensor>> = vec![TrajectoryBuffer::default(); num_envs];
        let (buffers, buffer_handlers) = bimodal_array(buffers);
        let worker_pool = match execution_mode {
            SamplerExecutionMode::SingleThreaded => {
                let workers: Vec<_> = buffer_handlers
                    .into_iter()
                    .enumerate()
                    .map(|(idx, element_handle)| {
                        let env = env_builder.build_idx(idx).unwrap();
                        Worker::new(env, element_handle)
                    })
                    .collect();
                WorkerPool::Vec(workers)
            }
            SamplerExecutionMode::MultiThreaded => {
                let env_builder = Arc::new(env_builder);
                let workers: Vec<_> = buffer_handlers
                    .into_iter()
                    .enumerate()
                    .map(|(idx, element_handle)| {
                        let (command_tx, command_rx) = crossbeam::channel::unbounded();
                        let (res_tx, res_rx) = crossbeam::channel::unbounded();
                        let env_builder = env_builder.clone();
                        let worker_seed = sample_u64();
                        let handle = std::thread::spawn(move || {
                            set_seed(worker_seed);
                            let env = env_builder.build_idx(idx).unwrap();
                            let worker = Worker::new(env, element_handle);
                            let mut thread_worker = ThreadWorker::new(worker, command_rx, res_tx);
                            thread_worker.work();
                        });
                        ThreadHandle::new(handle, command_tx, res_rx)
                    })
                    .collect();
                WorkerPool::Thread(ThreadWorkers::new(workers))
            }
        };
        Self {
            buffers,
            worker_pool,
        }
    }
}

/// Rollout sampler whose workers write directly to output buffers.
pub struct DirectSampler<E: Env, H: DirectSamplerHook<E = E>> {
    core: DirectSamplerCore<E>,
    hook: H,
}

impl<E: Env, H: DirectSamplerHook<E = E>> DirectSampler<E, H> {
    pub fn new(core: DirectSamplerCore<E>, hook: H) -> Self {
        Self { core, hook }
    }

    /// Builds a raw sampler and its environment workers.
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        hook: H,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        Self {
            core: DirectSamplerCore::build(env_builder, execution_mode),
            hook,
        }
    }

    /// Builds a homogeneous sampler from a shared environment builder.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_envs` is zero.
    pub fn build_from_env_builder(
        env_builder: Arc<dyn EnvBuilder<Env = E>>,
        num_envs: usize,
        hook: H,
        execution_mode: SamplerExecutionMode,
    ) -> Result<Self>
    where
        E: 'static,
    {
        let env_builder = move || env_builder.build_env();
        Ok(Self::build(
            EnvBuilderType::homogeneous(env_builder, num_envs)?,
            hook,
            execution_mode,
        ))
    }
}

impl<E: Env, H: DirectSamplerHook<E = E>> Sampler for DirectSampler<E, H> {
    type Tensor = E::Tensor;

    fn reset_all_envs(&mut self) -> Result<()> {
        self.core.reset_all_envs()?;
        self.hook.reset();
        Ok(())
    }

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> Result<()> {
        self.core.worker_pool.clear_buffers();
        self.core.worker_pool.set_actor(&actor);
        loop {
            let result = self.hook.hook(&mut self.core);
            match result {
                SamplerHookResult::Bound(bound) => self.core.worker_pool.collect(bound)?,
                SamplerHookResult::Stop => break,
            }
        }
        Ok(())
    }

    fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, Self::Tensor>]> {
        self.core
            .buffers
            .lock_map(|buffer| buffer.to_trajectory_view())
            .unwrap()
    }

    fn shutdown(&mut self) {
        self.core.worker_pool.shutdown();
    }
}
