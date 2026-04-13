pub mod worker;

use std::marker::PhantomData;
use std::sync::Arc;

use bimodal_array::ArrayHandle;
use bimodal_array::bimodal_array;
use r2l_core::buffers::ExpandableTrajectoryContainer;
use r2l_core::buffers::TrajectoryContainer;
use r2l_core::buffers::fix_sized::FixedSizeStateBuffer;
use r2l_core::buffers::variable_sized::VariableSizedStateBuffer;
use r2l_core::distributions::Actor;
use r2l_core::env::Env;
use r2l_core::env::EnvironmentDescription;
use r2l_core::env_builder::EnvBuilder;
use r2l_core::env_builder::EnvBuilderTrait;
use r2l_core::on_policy_algorithm::Sampler;
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

pub struct StepTrajectoryBound<T: R2lTensor> {
    steps: usize,
    _phantom: PhantomData<T>,
}

impl<T: R2lTensor> StepTrajectoryBound<T> {
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

pub struct EpisodeTrajectoryBound<T: R2lTensor> {
    episodes: usize,
    _phantom: PhantomData<T>,
}

impl<T: R2lTensor> EpisodeTrajectoryBound<T> {
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

pub enum Location {
    Vec,
    Thread,
}

pub trait PreprocessorY<T: R2lTensor, B: TrajectoryContainer<Tensor = T>> {
    // The question is, can we make this dyn compatible? Otherwise we just use a ref
    fn preprocess_states(&mut self, policy: &dyn Actor<Tensor = T>, buffers: &mut [B]);
}

// BD: collection method should probably be an enum!
pub struct FinalSampler<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> {
    all_buffers: ArrayHandle<BD::Container>,
    worker_pool: WorkerPool<E, BD::Container>,
    rollout_mode: RolloutMode,
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> FinalSampler<E, BD> {
    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilder<EB>,
        collection_method: BD,
        location: Location,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = (0..num_envs)
            .map(|_| collection_method.to_container())
            .collect();
        let (all_buffers, buffer_handlers) = bimodal_array(buffers);
        let worker_pool = match location {
            Location::Vec => {
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
            Location::Thread => {
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

    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        self.worker_pool.env_description()
    }
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> Sampler for FinalSampler<E, BD> {
    type Tensor = E::Tensor;
    type TrajectoryContainer = BD::Container;

    fn collect_rollouts<P: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) -> impl AsRef<[Self::TrajectoryContainer]> {
        self.worker_pool.set_policy(policy.clone());
        let rollout_mode = self.rollout_mode;
        self.worker_pool.collect(rollout_mode);
        self.all_buffers.lock().unwrap()
    }

    fn shutdown(&mut self) {
        self.worker_pool.shutdown();
    }
}
