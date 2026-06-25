// Here workers do not write to a buffer. Instead they return stuff directly to the sampler

use std::{marker::PhantomData, thread::JoinHandle};

use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, MultiMemory},
    env::{Env, EnvBuilder, EnvBuilderType, Snapshot},
    models::Actor,
    rng::RNG,
    tensor::R2lTensor,
};
use rand::RngExt;

pub struct Worker<E: Env<Tensor: R2lTensor>> {
    last_state: Option<E::Tensor>,
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<E: Env<Tensor: R2lTensor>> Worker<E> {
    pub fn from_env(env: E) -> Self {
        Self {
            last_state: None,
            actor: None,
            env,
        }
    }

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

pub enum WorkerCommand<T: R2lTensor> {
    Step(PhantomData<T>),
    SetPolicy(Box<dyn Actor<Tensor = T>>),
}

pub enum WorkerResult<T: R2lTensor> {
    Stepped(Memory<T>),
    PolicySet,
}

type CommandReceiver<T> = Receiver<WorkerCommand<T>>;
type CommandSender<T> = Sender<WorkerCommand<T>>;

type ResultReceiver<T> = Receiver<WorkerResult<T>>;
type ResultSender<T> = Sender<WorkerResult<T>>;

pub struct ThreadWorker<E: Env<Tensor: R2lTensor>> {
    worker: Worker<E>,
    command_receiver: CommandReceiver<E::Tensor>,
    result_sender: ResultSender<E::Tensor>,
}

impl<E: Env<Tensor: R2lTensor>> ThreadWorker<E> {
    pub fn new(
        worker: Worker<E>,
        command_receiver: CommandReceiver<E::Tensor>,
        result_sender: ResultSender<E::Tensor>,
    ) -> Self {
        Self {
            worker,
            command_receiver,
            result_sender,
        }
    }

    fn step(&mut self) {
        let memory = self.worker.step();
        self.result_sender
            .send(WorkerResult::Stepped(memory))
            .unwrap();
    }

    pub fn work(&mut self) {
        while let Ok(command) = self.command_receiver.recv() {
            match command {
                WorkerCommand::Step(_) => self.step(),
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.actor = Some(policy);
                    self.result_sender.send(WorkerResult::PolicySet).unwrap();
                }
            }
        }
    }
}

pub struct ThreadWorkers<T: R2lTensor> {
    worker_handles: Vec<ThreadHandle<T>>,
}

impl<T: R2lTensor> ThreadWorkers<T> {
    pub fn new(worker_handles: Vec<ThreadHandle<T>>) -> Self {
        Self { worker_handles }
    }

    fn step(&self) -> MultiMemory<T> {
        let mut multi_memory = MultiMemory::with_capacity(self.worker_handles.len());
        for worker in &self.worker_handles {
            worker
                .command_sender
                .send(WorkerCommand::Step(PhantomData))
                .unwrap();
        }
        for worker in &self.worker_handles {
            let Ok(WorkerResult::Stepped(memory)) = worker.result_receiver.recv() else {
                // should be unreachable I guess
                todo!()
            };
            multi_memory.push_memory(memory);
        }
        multi_memory
    }

    pub fn set_policy<A: Actor<Tensor = T> + Clone>(&self, policy: A) {
        for worker in self.worker_handles.iter() {
            worker
                .command_sender
                .send(WorkerCommand::SetPolicy(Box::new(policy.clone())));
        }
        for worker in self.worker_handles.iter() {
            worker.result_receiver.recv();
        }
    }
}

pub struct ThreadHandle<T: R2lTensor> {
    handle: JoinHandle<()>,
    command_sender: CommandSender<T>,
    result_receiver: ResultReceiver<T>,
}

impl<T: R2lTensor> ThreadHandle<T> {
    pub fn new(
        handle: JoinHandle<()>,
        command_sender: CommandSender<T>,
        result_receiver: ResultReceiver<T>,
    ) -> Self {
        Self {
            handle,
            command_sender,
            result_receiver,
        }
    }
}

pub struct VecWorkers<E: Env<Tensor: R2lTensor>> {
    workers: Vec<Worker<E>>,
}

impl<E: Env<Tensor: R2lTensor>> VecWorkers<E> {
    pub fn from_env_builder<EB: EnvBuilder<Env = E>>(env_builder: EnvBuilderType<EB>) -> Self {
        let num_envs = env_builder.num_envs();
        let workers = (0..num_envs)
            .map(|idx| {
                let env = env_builder.build_idx(idx).unwrap();
                Worker::from_env(env)
            })
            .collect();
        Self { workers }
    }

    fn step(&mut self) -> MultiMemory<E::Tensor> {
        let mut multi_memory = MultiMemory::with_capacity(self.workers.len());
        for worker in &mut self.workers {
            let memory = worker.step();
            multi_memory.push_memory(memory);
        }
        multi_memory
    }
}

pub enum WorkerPool<E: Env<Tensor: R2lTensor>> {
    VecCoord(VecWorkers<E>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env<Tensor: R2lTensor>> WorkerPool<E> {
    pub fn step(&mut self) -> MultiMemory<E::Tensor> {
        match self {
            Self::Thread(threads) => threads.step(),
            Self::VecCoord(workers) => workers.step(),
        }
    }

    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        match self {
            Self::VecCoord(vec_workers) => {
                for worker in vec_workers.workers.iter_mut() {
                    worker.actor = Some(Box::new(policy.clone()))
                }
            }
            Self::Thread(thread_workers) => {
                thread_workers.set_policy(policy);
            }
        }
    }
}
