// The final design

pub mod buffer;
pub mod worker;

use crate::{env::EnvironmentDescription, rng::RNG, sampler5::worker::Worker};
use std::sync::Arc;

use bimodal_array::ArrayHandle;
use rand::Rng;

use crate::{
    distributions::Policy,
    env::Env,
    env_builder::{EnvBuilderTrait, EnvBuilderType},
    sampler5::{
        buffer::{TrajectoryBound, TrajectoryContainer},
        worker::{ThreadWorker, ThreadWorkers, WorkerPool},
    },
    tensor::R2lTensor,
};
use bimodal_array::bimodal_array;

#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}

pub trait Sampler5 {
    type Tensor: R2lTensor;
    type TrajectoryContainer: TrajectoryContainer<Tensor = Self::Tensor>;

    fn collect_rollouts<P: Policy<Tensor = Self::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) -> impl AsRef<[Self::TrajectoryContainer]>;
}

pub enum Location {
    Vec,
    Thread,
}

pub trait PreprocessorY<B: TrajectoryContainer> {
    // The question is, can we make this dyn compatible? Otherwise we just use a ref
    fn preprocess_states(&mut self, buffers: &[B]);
}

pub struct FinalSampler<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> {
    preprocessor: Option<Box<dyn PreprocessorY<BD::Container>>>,
    all_buffers: ArrayHandle<BD::Container>,
    worker_pool: WorkerPool<E, BD::Container>,
    collection_method: BD,
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> FinalSampler<E, BD> {
    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        collection_method: BD,
        preprocessor: Option<Box<dyn PreprocessorY<BD::Container>>>,
        location: Location,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = (0..num_envs)
            .map(|_| collection_method.to_container())
            .collect();
        let (array_handle, element_handles) = bimodal_array(buffers);
        let worker_pool = match location {
            Location::Vec => {
                let workers = element_handles
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
                let workers = element_handles
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
            preprocessor,
            all_buffers: array_handle,
            worker_pool,
            collection_method,
        }
    }

    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        self.worker_pool.env_description()
    }
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> Sampler5 for FinalSampler<E, BD> {
    type Tensor = E::Tensor;
    type TrajectoryContainer = BD::Container;

    fn collect_rollouts<P: Policy<Tensor = Self::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) -> impl AsRef<[Self::TrajectoryContainer]> {
        self.worker_pool.set_policy(policy);
        let bound = self.collection_method.method();
        if let Some(pre_processor) = &mut self.preprocessor {
            let mut current_step = 0;
            // TODO: with the new implementation, this might became trivial
            let RolloutMode::StepBound { n_steps: steps } = bound else {
                panic!("pre processors currently only support rollout bounds");
            };
            while current_step < steps {
                let buffers = self.all_buffers.lock().unwrap();
                pre_processor.preprocess_states(buffers.as_ref());
                drop(buffers);
                self.worker_pool.single_step();
                current_step += 1;
            }
        } else {
            self.worker_pool.collect(bound);
        }
        self.all_buffers.lock().unwrap()
    }
}
