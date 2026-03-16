pub mod buffer;
pub mod workers;

use crate::{
    distributions::Policy,
    env::{Env, RolloutMode},
    env_builder::{EnvBuilderTrait, EnvBuilderType},
    sampler4::{
        buffer::{BufferS, BufferS2, TrajectoryBound, TrajectoryContainer},
        workers::{ThreadWorker, ThreadWorkers, Worker, WorkerPool},
    },
    tensor::R2lTensor,
};
use anyhow::Result;
use bimodal_array::{ArrayHandle, bimodal_array};
use std::{marker::PhantomData, sync::Arc};

pub trait Sampler3 {
    type Tensor: R2lTensor;
    type Env: Env<Tensor = Self::Tensor>;
    type Buffer: TrajectoryContainer<Tensor = Self::Tensor>;

    fn collect_rollouts<P: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) -> impl AsRef<[BufferS<Self::Buffer>]>;
}

#[derive(Debug, Clone)]
pub struct PolicyWrapper<P: Policy + Clone, T: R2lTensor> {
    policy: P,
    env: PhantomData<T>,
}

impl<D: Policy + Clone, T: R2lTensor> PolicyWrapper<D, T> {
    pub fn new(policy: D) -> Self {
        Self {
            policy,
            env: PhantomData,
        }
    }
}

impl<D: Policy + Clone, T: R2lTensor> Policy for PolicyWrapper<D, T>
where
    T: From<D::Tensor>,
    T: Into<D::Tensor>,
{
    type Tensor = T;

    fn std(&self) -> Result<f32> {
        self.policy.std()
    }

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.policy.get_action(observation.into())?;
        Ok(action.into())
    }

    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor> {
        let observations = observations
            .iter()
            .map(|o| o.clone().into())
            .collect::<Vec<_>>();
        let actions = actions.iter().map(|a| a.clone().into()).collect::<Vec<_>>();
        let log_probs = self.policy.log_probs(&observations, &actions)?;
        Ok(log_probs.into())
    }

    fn entropy(&self) -> Result<Self::Tensor> {
        let entropy = self.policy.entropy()?;
        Ok(entropy.into())
    }

    fn resample_noise(&mut self) -> Result<()> {
        // TODO: we may want the distribution to be behind a RwLock, but I doubt that this will be
        // called a whole lot. In future releases we should enable finer control of noise sampling
        todo!()
    }
}

pub enum Location {
    Vec,
    Thread,
}

pub trait PreprocessorY<B: TrajectoryContainer> {
    // The question is, can we make this dyn compatible? Otherwise we just use a ref
    fn preprocess_states(&mut self, buffers: &[BufferS<B>]);
}

pub struct NewSampler<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> {
    preprocessor: Option<Box<dyn PreprocessorY<BD::Container>>>,
    all_buffers: ArrayHandle<BufferS<BD::Container>>,
    worker_pool: WorkerPool<E, BD::Container>,
    collection_method: BD,
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> NewSampler<E, BD> {
    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        collection_method: BD,
        preprocessor: Option<Box<dyn PreprocessorY<BD::Container>>>,
        location: Location,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = (0..num_envs)
            .map(|_| {
                let container = collection_method.to_container();
                BufferS::new(container)
            })
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
}

impl<E: Env, BD: TrajectoryBound<Tensor = E::Tensor>> Sampler3 for NewSampler<E, BD> {
    type Tensor = E::Tensor;
    type Env = E;
    type Buffer = BD::Container;

    fn collect_rollouts<P: Policy<Tensor = Self::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) -> impl AsRef<[BufferS<Self::Buffer>]> {
        let policy = PolicyWrapper::new(policy);
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

pub trait Sampler4 {
    type Tensor: R2lTensor;

    fn collect_rollouts<P: Policy<Tensor = Self::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) -> impl AsRef<[BufferS2<Self::Tensor>]>;
}
