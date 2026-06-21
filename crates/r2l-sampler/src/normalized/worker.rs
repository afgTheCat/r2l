// Here workers do not write to a buffer. Instead they return stuff directly to the sampler

use std::{marker::PhantomData, thread::JoinHandle};

use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    env::{Env, Snapshot},
    models::Actor,
    rng::RNG,
    tensor::RunningMeanTensor,
};
use rand::RngExt;

pub struct Worker<E: Env<Tensor: RunningMeanTensor>> {
    last_state: Option<E::Tensor>,
    policy: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<E: Env<Tensor: RunningMeanTensor>> Worker<E> {
    // state, next_state,
    fn step(&mut self) -> (E::Tensor, f32, bool) {
        let Some(policy) = &mut self.policy else {
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
            mut state,
            reward,
            terminated,
            truncated,
        } = self.env.step(action).unwrap();
        let done = terminated || truncated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            state = self.env.reset(seed).unwrap();
        }
        self.last_state = Some(state.clone());
        (state, reward, done)
    }
}

enum WorkerCommand<T: RunningMeanTensor> {
    Step(PhantomData<T>),
}

enum WorkerResult<T: RunningMeanTensor> {
    Stepped((T, f32, bool)),
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
        let step_res = self.worker.step();
        self.result_sender
            .send(WorkerResult::Stepped(step_res))
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
    fn step(&self) -> (Vec<T>, Vec<f32>, Vec<bool>) {
        let mut states = Vec::with_capacity(self.worker_handles.len());
        let mut rewards = Vec::with_capacity(self.worker_handles.len());
        let mut dones = Vec::with_capacity(self.worker_handles.len());
        for worker in &self.worker_handles {
            worker
                .command_sender
                .send(WorkerCommand::Step(PhantomData))
                .unwrap();
        }
        for worker in &self.worker_handles {
            let WorkerResult::Stepped((state, reward, done)) =
                worker.result_receiver.recv().unwrap();
            states.push(state);
            rewards.push(reward);
            dones.push(done);
        }
        (states, rewards, dones)
    }
}

pub struct ThreadHandle<T: RunningMeanTensor> {
    handle: JoinHandle<()>,
    command_sender: CommandSender<T>,
    result_receiver: ResultReceiver<T>,
}

pub enum WorkerPool<E: Env<Tensor: RunningMeanTensor>> {
    VecCoord(Vec<Worker<E>>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env<Tensor: RunningMeanTensor>> WorkerPool<E> {
    pub fn step(&mut self) -> (Vec<E::Tensor>, Vec<f32>, Vec<bool>) {
        match self {
            Self::Thread(threads) => threads.step(),
            Self::VecCoord(workers) => {
                let mut states = Vec::with_capacity(workers.len());
                let mut rewards = Vec::with_capacity(workers.len());
                let mut dones = Vec::with_capacity(workers.len());
                for worker in workers {
                    let (state, reward, done) = worker.step();
                    states.push(state);
                    rewards.push(reward);
                    dones.push(done);
                }
                (states, rewards, dones)
            }
        }
    }
}
