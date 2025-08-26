use crate::{
    distributions::Distribution,
    env::Env,
    numeric::Buffer,
    sampler::{
        env_pools::FixedSizeEnvPool,
        trajectory_buffers::fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
    },
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;
use crossbeam::{
    channel::{Receiver, Sender},
    sync::ShardedLock,
};
use std::sync::Arc;

enum WorkerTask<E: Env> {
    SingleStep { thread_id: usize },
    SetBuffer { buffer: FixedSizeStateBuffer<E> },
}

enum WorkerResult<E: Env> {
    Step {
        thread_id: usize,
        buffer: FixedSizeStateBuffer<E>,
    },
}

struct WorkerThread<E: Env> {
    tx: Sender<WorkerResult<E>>,
    rx: Receiver<WorkerTask<E>>,
    buffer: FixedSizeTrajectoryBuffer<E>,
}

struct FixedSizeThreadEnvPool<E: Env> {
    tx: Vec<Sender<WorkerTask<E>>>,
    rx: Vec<Receiver<WorkerResult<E>>>,
    // TODO: This is bad and ugly maybe we shoild get rid of it
    distr_lock: Arc<ShardedLock<Option<&'static dyn Distribution<Tensor = Tensor>>>>,
}

impl<E: Env<Tensor = Buffer>> FixedSizeEnvPool for FixedSizeThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.tx.len()
    }

    fn single_step<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        todo!()
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        todo!()
    }

    fn to_rollout_buffers(&mut self, steps_per_environment: usize) -> Vec<RolloutBuffer> {
        todo!()
    }

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        todo!()
    }
}
