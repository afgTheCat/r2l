// pub mod sampler;
pub mod worker;

use std::sync::Arc;

use bimodal_array::ArrayHandle;
use bimodal_array::bimodal_array;
use r2l_core::buffers::buffer::TrajectoryBuffer;
use r2l_core::buffers::buffer::TrajectoryView;
use r2l_core::env::Env;
use r2l_core::env::EnvBuilder;
use r2l_core::env::EnvBuilderType;
use r2l_core::models::Actor;
use r2l_core::on_policy::algorithm::Sampler;
use r2l_core::running_mean::RunningMeanStd2;
use r2l_core::tensor::RunningMeanTensor;

use crate::worker::ThreadHandle;
use crate::worker::ThreadWorker;
use crate::worker::ThreadWorkers;
use crate::worker::Worker;
use crate::worker::WorkerPool;

#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
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

pub enum SamplerHookResult {
    Stop,
    Bound(RolloutMode),
}

pub trait SamplerHook {
    type E: Env;

    fn hook(
        &mut self,
        buffer: &mut ArrayHandle<TrajectoryBuffer<<Self::E as Env>::Tensor>>,
        worker_pool: &mut WorkerPool<Self::E>,
    ) -> SamplerHookResult;

    fn observation_normalizer(&self) -> Option<RunningMeanStd2<<Self::E as Env>::Tensor>>
    where
        <Self::E as Env>::Tensor: RunningMeanTensor,
    {
        None
    }
}

pub struct R2lSampler<E: Env, H: SamplerHook<E = E>> {
    buffers: ArrayHandle<TrajectoryBuffer<E::Tensor>>,
    worker_pool: WorkerPool<E>,
    hook: H,
}

impl<E: Env, H: SamplerHook<E = E>> R2lSampler<E, H> {
    pub fn reset_all_envs(&mut self) {
        self.worker_pool.reset_all_envs();
    }

    pub fn to_views(&mut self) -> impl AsRef<[TrajectoryView<'_, E::Tensor>]> {
        self.buffers
            .lock_map(|buffer| buffer.to_trajectory_view())
            .unwrap()
    }

    pub fn observation_normalizer(&self) -> Option<RunningMeanStd2<E::Tensor>>
    where
        E::Tensor: RunningMeanTensor,
    {
        self.hook.observation_normalizer()
    }

    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        hook: H,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        // questionable if we want to do this, but whatever
        let num_envs = env_builder.num_envs();
        let buffers: Vec<TrajectoryBuffer<E::Tensor>> = vec![TrajectoryBuffer::default(); num_envs];
        let (buffers, buffer_handlers) = bimodal_array(buffers);
        let worker_pool = match execution_mode {
            SamplerExecutionMode::Vec => {
                let workers: Vec<_> = buffer_handlers
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
                let workers: Vec<_> = buffer_handlers
                    .into_iter()
                    .enumerate()
                    .map(|(idx, element_handle)| {
                        let (command_tx, command_rx) = crossbeam::channel::unbounded();
                        let (res_tx, res_rx) = crossbeam::channel::unbounded();
                        let env_builder = env_builder.clone();
                        let handle = std::thread::spawn(move || {
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
            hook,
        }
    }
}

impl<E: Env, H: SamplerHook<E = E>> Sampler for R2lSampler<E, H> {
    type Tensor = E::Tensor;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        self.worker_pool.clear_buffers();
        self.worker_pool.set_policy(actor.clone());
        loop {
            let result = self.hook.hook(&mut self.buffers, &mut self.worker_pool);
            match result {
                SamplerHookResult::Bound(bound) => self.worker_pool.collect(bound),
                SamplerHookResult::Stop => break,
            }
        }
    }

    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]> {
        self.buffers
            .lock_map(|buffer| buffer.to_trajectory_view())
            .unwrap()
    }

    fn observation_normalizer(&self) -> Option<RunningMeanStd2<Self::Tensor>>
    where
        Self::Tensor: RunningMeanTensor,
    {
        self.hook.observation_normalizer()
    }

    fn shutdown(&mut self) {
        self.worker_pool.shutdown();
    }
}
