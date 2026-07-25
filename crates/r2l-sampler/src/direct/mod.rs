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

pub enum SamplerHookResult {
    Stop,
    Bound(RolloutMode),
}

pub trait SamplerHook {
    type E: Env;

    fn hook(&mut self, core: &mut R2lSamplerCore<Self::E>) -> SamplerHookResult;

    fn reset(&mut self) {}
}

pub struct R2lSamplerCore<E: Env> {
    pub buffers: ArrayHandle<TrajectoryBuffer<E::Tensor>>,
    pub worker_pool: WorkerPool<E>,
}

impl<E: Env> R2lSamplerCore<E> {
    pub fn reset_all_envs(&mut self) {
        self.worker_pool.reset_all_envs();
    }

    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers: Vec<TrajectoryBuffer<E::Tensor>> = vec![TrajectoryBuffer::default(); num_envs];
        let (buffers, buffer_handlers) = bimodal_array(buffers);
        let worker_pool = match execution_mode {
            SamplerExecutionMode::Vec => {
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
            SamplerExecutionMode::Thread => {
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

pub struct R2lSampler<E: Env, H: SamplerHook<E = E>> {
    core: R2lSamplerCore<E>,
    hook: H,
}

impl<E: Env, H: SamplerHook<E = E>> R2lSampler<E, H> {
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        hook: H,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        Self {
            core: R2lSamplerCore::build(env_builder, execution_mode),
            hook,
        }
    }
}

impl<E: Env, H: SamplerHook<E = E>> Sampler for R2lSampler<E, H> {
    type Tensor = E::Tensor;

    fn reset_all_envs(&mut self) {
        self.core.reset_all_envs();
        self.hook.reset();
    }

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        self.core.worker_pool.clear_buffers();
        self.core.worker_pool.set_actor(actor.clone());
        loop {
            let result = self.hook.hook(&mut self.core);
            match result {
                SamplerHookResult::Bound(bound) => self.core.worker_pool.collect(bound),
                SamplerHookResult::Stop => break,
            }
        }
    }

    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]> {
        self.core
            .buffers
            .lock_map(|buffer| buffer.to_trajectory_view())
            .unwrap()
    }

    fn shutdown(&mut self) {
        self.core.worker_pool.shutdown();
    }
}
