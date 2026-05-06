mod hooks;
pub mod worker;

use std::marker::PhantomData;
use std::sync::Arc;

use bimodal_array::ArrayHandle;
use bimodal_array::bimodal_array;
use r2l_core::buffers::ExpandableTrajectoryContainer;
use r2l_core::buffers::fix_sized::FixedSizeStateBuffer;
use r2l_core::buffers::variable_sized::VariableSizedStateBuffer;
use r2l_core::env::Env;
use r2l_core::env::EnvBuilder;
use r2l_core::env::EnvBuilderType;
use r2l_core::env::EnvDescription;
use r2l_core::models::Actor;
use r2l_core::on_policy::algorithm::Sampler;
use r2l_core::tensor::R2lTensor;

use crate::worker::ThreadWorker;
use crate::worker::ThreadWorkers;
use crate::worker::Worker;
use crate::worker::WorkerPool;

#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}

pub trait TrajectoryBound: Send + Sync {
    type Tensor: R2lTensor;
    // The caontainer type that is able to work with the given trajectory bound
    type Container: ExpandableTrajectoryContainer<Tensor = Self::Tensor>;

    fn to_container(&self) -> Self::Container;
    fn method(&self) -> RolloutMode;
}

/// Trajectory bound that yields rollouts after a fixed number of steps.
///
/// This bound uses a fixed-size trajectory container and stops collection once
/// each sampler worker has produced `steps` transitions for the current
/// rollout.
pub struct StepTrajectoryBound<T: R2lTensor> {
    steps: usize,
    _phantom: PhantomData<T>,
}

impl<T: R2lTensor> StepTrajectoryBound<T> {
    /// Creates a step-bounded trajectory configuration.
    pub fn new(steps: usize) -> Self {
        Self {
            steps,
            _phantom: PhantomData,
        }
    }
}

impl<T: R2lTensor> TrajectoryBound for StepTrajectoryBound<T> {
    type Tensor = T;
    type Container = FixedSizeStateBuffer<T>;

    fn to_container(&self) -> Self::Container {
        FixedSizeStateBuffer::new(self.steps)
    }

    fn method(&self) -> RolloutMode {
        RolloutMode::StepBound {
            n_steps: self.steps,
        }
    }
}

/// Trajectory bound that yields rollouts after a fixed number of episodes.
///
/// This bound uses a variable-sized trajectory container and stops collection
/// once each sampler worker has completed `episodes` full episodes for the
/// current rollout.
pub struct EpisodeTrajectoryBound<T: R2lTensor> {
    episodes: usize,
    _phantom: PhantomData<T>,
}

impl<T: R2lTensor> EpisodeTrajectoryBound<T> {
    /// Creates an episode-bounded trajectory configuration.
    pub fn new(episodes: usize) -> Self {
        Self {
            episodes,
            _phantom: PhantomData,
        }
    }
}

impl<T: R2lTensor> TrajectoryBound for EpisodeTrajectoryBound<T> {
    type Tensor = T;
    type Container = VariableSizedStateBuffer<T>;

    fn to_container(&self) -> Self::Container {
        VariableSizedStateBuffer::new()
    }

    fn method(&self) -> RolloutMode {
        RolloutMode::EpisodeBound {
            n_episodes: self.episodes,
        }
    }
}

/// Execution strategy used by [`R2lSampler`] workers.
///
/// This controls whether environment workers run inline in the current thread
/// or in dedicated background threads.
pub enum SamplerExecutionMode {
    /// Run sampler workers inline in a local vector on the current thread.
    Vec,
    /// Run sampler workers in dedicated background threads.
    Thread,
}

/// Vectorized rollout sampler used by on-policy algorithms.
///
/// `R2lSampler` owns a set of environment workers together with one trajectory
/// container per worker. On each rollout collection, it pushes the current
/// policy to all workers, steps the environments according to the configured
/// [`RolloutMode`], and returns the collected trajectories.
///
/// The sampler is parameterized by:
/// - `E`: the environment type being sampled
/// - `BD`: the [`TrajectoryBound`] that decides when rollout collection stops
///   and which trajectory container implementation is used
///
/// Instances are typically constructed through `r2l_api::SamplerBuilder` or by
/// higher-level algorithm builders in `r2l-api`.
// ANCHOR: r2l_sampler
pub struct R2lSampler<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> {
    all_buffers: ArrayHandle<BD::Container>,
    worker_pool: WorkerPool<E, BD::Container>,
    rollout_mode: RolloutMode,
}
// ANCHOR_END: r2l_sampler

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> R2lSampler<E, BD> {
    /// Builds a sampler from an environment builder, trajectory bound, and
    /// execution mode.
    ///
    /// `collection_method` determines both the rollout stopping condition and
    /// the trajectory container type allocated for each environment worker.
    /// `location` controls whether workers run inline or on background threads.
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        collection_method: BD,
        location: SamplerExecutionMode,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = (0..num_envs)
            .map(|_| collection_method.to_container())
            .collect();
        let (all_buffers, buffer_handlers) = bimodal_array(buffers);
        let worker_pool = match location {
            SamplerExecutionMode::Vec => {
                let workers = buffer_handlers
                    .into_iter()
                    .enumerate()
                    .map(|(idx, element_handle)| {
                        let env = env_builder.build_idx(idx).unwrap(); // TODO: for now
                        Worker::new(env, element_handle)
                    })
                    .collect();
                WorkerPool::Vec(workers)
            }
            SamplerExecutionMode::Thread => {
                let env_builder = Arc::new(env_builder);
                let workers = buffer_handlers
                    .into_iter()
                    .enumerate()
                    .map(|(idx, element_handle)| {
                        let (command_tx, command_rx) = crossbeam::channel::unbounded();
                        let (res_tx, res_rx) = crossbeam::channel::unbounded();
                        let env_builder = env_builder.clone();
                        std::thread::spawn(move || {
                            let env = env_builder.build_idx(idx).unwrap();
                            let worker = Worker::new(env, element_handle);
                            let mut thread_worker = ThreadWorker::new(worker, command_rx, res_tx);
                            thread_worker.work();
                        });
                        (idx, (command_tx, res_rx))
                    })
                    .collect();
                WorkerPool::Thread(ThreadWorkers(workers))
            }
        };
        Self {
            all_buffers,
            worker_pool,
            rollout_mode: collection_method.method(),
        }
    }

    /// Returns the environment description shared by this sampler's workers.
    pub fn env_description(&self) -> EnvDescription<E::Tensor> {
        self.worker_pool.env_description()
    }
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> Sampler for R2lSampler<E, BD> {
    type Tensor = E::Tensor;
    type TrajectoryContainer = BD::Container;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> impl AsRef<[Self::TrajectoryContainer]> {
        self.worker_pool.set_policy(actor.clone());
        let rollout_mode = self.rollout_mode;
        self.worker_pool.collect(rollout_mode);
        self.all_buffers.lock().unwrap()
    }

    fn shutdown(&mut self) {
        self.worker_pool.shutdown();
    }
}
