use crate::{
    buffers::episode_bound_buffer::{StateBuffer, StepBoundBuffer},
    distributions::Distribution,
    env::Env,
    numeric::Buffer,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;
use crossbeam::{
    channel::{Receiver, Sender},
    sync::ShardedLock,
};
use std::sync::Arc;

pub trait FixedSizeEnvPool {
    type Env: Env<Tensor = Buffer>;

    fn num_envs(&self) -> usize;

    fn single_step_and_collect<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<StateBuffer<Self::Env>>;

    fn set_buffers(&mut self, buffers: Vec<StateBuffer<Self::Env>>);

    // TODO: should be removed once we have a trait for the trajectory buffers
    fn to_rollout_buffers(&mut self, steps_per_environment: usize) -> Vec<RolloutBuffer>;

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize);
}

pub struct FixedSizeVecEnvPool<E: Env> {
    buffers: Vec<StepBoundBuffer<E>>,
}

impl<E: Env<Tensor = Buffer>> FixedSizeEnvPool for FixedSizeVecEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.buffers.len()
    }

    // TODO: a more flexible environment would be one that can take a function
    fn single_step_and_collect<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<StateBuffer<E>> {
        let mut state_buffers = vec![];
        for buf in self.buffers.iter_mut() {
            buf.step(distr);
            state_buffers.push(buf.move_buffer());
        }
        state_buffers
    }

    fn set_buffers(&mut self, buffers: Vec<StateBuffer<E>>) {
        for (buf, state_buffer) in self.buffers.iter_mut().zip(buffers) {
            buf.set_buffer(state_buffer);
        }
    }

    fn to_rollout_buffers(&mut self, steps_per_environment: usize) -> Vec<RolloutBuffer> {
        self.buffers
            .iter_mut()
            .map(|buff| {
                buff.buffer
                    .as_mut()
                    .unwrap()
                    .to_rollout_buffer(steps_per_environment)
            })
            .collect()
    }

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        for buf in self.buffers.iter_mut() {
            for _ in 0..steps {
                buf.step(distr);
            }
        }
    }
}

enum WorkerTask<E: Env> {
    SingleStep { thread_id: usize },
    SetBuffer { buffer: StateBuffer<E> },
}

enum WorkerResult<E: Env> {
    Step {
        thread_id: usize,
        buffer: StateBuffer<E>,
    },
}

struct WorkerThread<E: Env> {
    tx: Sender<WorkerResult<E>>,
    rx: Receiver<WorkerTask<E>>,
    buffer: StepBoundBuffer<E>,
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

    fn single_step_and_collect<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<StateBuffer<Self::Env>> {
        todo!()
    }

    fn set_buffers(&mut self, buffers: Vec<StateBuffer<Self::Env>>) {
        todo!()
    }

    fn to_rollout_buffers(&mut self, steps_per_environment: usize) -> Vec<RolloutBuffer> {
        todo!()
    }

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        todo!()
    }
}
