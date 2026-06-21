// Here workers do not write to a buffer. Instead they return stuff directly to the sampler

use std::{marker::PhantomData, thread::JoinHandle};

use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, MultiMemory},
    env::{Env, Snapshot},
    models::Actor,
    rng::RNG,
    tensor::RunningMeanTensor,
};
use rand::RngExt;

pub struct Worker<E: Env<Tensor: RunningMeanTensor>> {
    last_state: Option<E::Tensor>,
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<E: Env<Tensor: RunningMeanTensor>> Worker<E> {
    // state, next_state,
    fn step(&mut self) -> Memory<E::Tensor> {
        let Some(policy) = &mut self.actor else {
            todo!()
        };
        let state = if let Some(state) = self.last_state.take() {
            state
        } else {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            self.env.reset(seed).unwrap()
        };
        let action = policy.action(state.clone()).unwrap();
        let Snapshot {
            state: mut next_state,
            reward,
            terminated,
            truncated,
        } = self.env.step(action.clone()).unwrap();
        let done = terminated || truncated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            next_state = self.env.reset(seed).unwrap();
        }
        self.last_state = Some(next_state.clone());
        Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        }
    }
}

enum WorkerCommand<T: RunningMeanTensor> {
    Step(PhantomData<T>),
}

enum WorkerResult<T: RunningMeanTensor> {
    Stepped(Memory<T>),
}

type CommandReceiver<T> = Receiver<WorkerCommand<T>>;
type CommandSender<T> = Sender<WorkerCommand<T>>;

type ResultReceiver<T> = Receiver<WorkerResult<T>>;
type ResultSender<T> = Sender<WorkerResult<T>>;

pub struct ThreadWorker<E: Env<Tensor: RunningMeanTensor>> {
    worker: Worker<E>,
    command_receiver: CommandReceiver<E::Tensor>,
    result_sender: ResultSender<E::Tensor>,
}

impl<E: Env<Tensor: RunningMeanTensor>> ThreadWorker<E> {
    fn step(&mut self) {
        let memory = self.worker.step();
        self.result_sender
            .send(WorkerResult::Stepped(memory))
            .unwrap();
    }

    fn work(&mut self) {
        while let Ok(command) = self.command_receiver.recv() {
            match command {
                WorkerCommand::Step(_) => self.step(),
            }
        }
    }
}

pub struct ThreadWorkers<T: RunningMeanTensor> {
    worker_handles: Vec<ThreadHandle<T>>,
}

impl<T: RunningMeanTensor> ThreadWorkers<T> {
    fn step(&self) -> MultiMemory<T> {
        let mut multi_memory = MultiMemory::with_capacity(self.worker_handles.len());
        for worker in &self.worker_handles {
            worker
                .command_sender
                .send(WorkerCommand::Step(PhantomData))
                .unwrap();
        }
        for worker in &self.worker_handles {
            let WorkerResult::Stepped(memory) = worker.result_receiver.recv().unwrap();
            multi_memory.push_memory(memory);
        }
        multi_memory
    }
}

pub struct ThreadHandle<T: RunningMeanTensor> {
    handle: JoinHandle<()>,
    command_sender: CommandSender<T>,
    result_receiver: ResultReceiver<T>,
}

struct VecWorkers<E: Env<Tensor: RunningMeanTensor>> {
    workers: Vec<Worker<E>>,
}

impl<E: Env<Tensor: RunningMeanTensor>> VecWorkers<E> {
    fn step(&mut self) -> MultiMemory<E::Tensor> {
        let mut multi_memory = MultiMemory::with_capacity(self.workers.len());
        for worker in &mut self.workers {
            let memory = worker.step();
            multi_memory.push_memory(memory);
        }
        multi_memory
    }
}

pub enum WorkerPool<E: Env<Tensor: RunningMeanTensor>> {
    VecCoord(VecWorkers<E>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env<Tensor: RunningMeanTensor>> WorkerPool<E> {
    pub fn step(&mut self) -> MultiMemory<E::Tensor> {
        match self {
            Self::Thread(threads) => threads.step(),
            Self::VecCoord(workers) => workers.step(),
        }
    }
}
