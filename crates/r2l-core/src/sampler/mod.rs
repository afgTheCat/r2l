pub mod buffer;
pub mod worker;

use crate::{
    distributions::Actor,
    env::Env,
    env_builder::{EnvBuilder, EnvBuilderTrait},
    sampler::{
        buffer::{TrajectoryBound, TrajectoryContainer},
        worker::{ThreadWorker, ThreadWorkers, WorkerPool},
    },
    tensor::R2lTensor,
};
use crate::{env::EnvironmentDescription, sampler::worker::Worker};
use anyhow::Result;
use bimodal_array::ArrayHandle;
use bimodal_array::bimodal_array;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}

pub trait Sampler {
    type Tensor: R2lTensor;
    type TrajectoryContainer: TrajectoryContainer<Tensor = Self::Tensor>;

    fn collect_rollouts<P: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) -> impl AsRef<[Self::TrajectoryContainer]>;

    fn shutdown(&mut self) {}
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
// TODO: this feels more like a builder to me, we will probably need a new version
pub struct FinalSampler<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> {
    preprocessor: Option<Box<dyn PreprocessorY<E::Tensor, BD::Container>>>,
    all_buffers: ArrayHandle<BD::Container>,
    worker_pool: WorkerPool<E, BD::Container>,
    rollout_mode: RolloutMode,
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> FinalSampler<E, BD> {
    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilder<EB>,
        collection_method: BD,
        preprocessor: Option<Box<dyn PreprocessorY<E::Tensor, BD::Container>>>,
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
            preprocessor,
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
        if let Some(pre_processor) = &mut self.preprocessor {
            let mut current_step = 0;
            // TODO: with the new implementation, this might became trivial
            let RolloutMode::StepBound { n_steps: steps } = rollout_mode else {
                panic!("pre processors currently only support rollout bounds");
            };
            while current_step < steps {
                let mut buffers = self.all_buffers.lock().unwrap();
                pre_processor.preprocess_states(&policy, buffers.as_mut());
                drop(buffers);
                self.worker_pool.single_step();
                current_step += 1;
            }
        } else {
            self.worker_pool.collect(rollout_mode);
        }
        self.all_buffers.lock().unwrap()
    }

    fn shutdown(&mut self) {
        self.worker_pool.shutdown();
    }
}

#[derive(Debug, Clone)]
pub struct ActorWrapper<A: Actor + Clone, T: R2lTensor> {
    policy: A,
    env: PhantomData<T>,
}

impl<D: Actor + Clone, T: R2lTensor> ActorWrapper<D, T> {
    pub fn new(policy: D) -> Self {
        Self {
            policy,
            env: PhantomData,
        }
    }
}

impl<D: Actor + Clone, T: R2lTensor> Actor for ActorWrapper<D, T>
where
    T: From<D::Tensor>,
    T: Into<D::Tensor>,
{
    type Tensor = T;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.policy.get_action(observation.into())?;
        Ok(action.into())
    }
}
