use bimodal_array::{ElementHandle, ElementWorker, ElementWorkerFactory};
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, MultiMemory},
    env::{Env, EnvBuilder, Snapshot},
    models::Actor,
    rng::RNG,
    tensor::R2lTensor,
};
use rand::RngExt;

pub enum WorkerCommand<T: R2lTensor> {
    Step,
    SetPolicy(Box<dyn Actor<Tensor = T>>),
    Stop,
}

pub enum WorkerResult<T: R2lTensor> {
    Stepped(Memory<T>),
    PolicySet,
    Stopped,
}

struct Worker<T: R2lTensor, E: Env<Tensor = T>> {
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<T: R2lTensor, E: Env<Tensor = T>> Worker<T, E> {
    fn new(env: E) -> Self {
        Self { actor: None, env }
    }

    fn step(&mut self, handle: &mut ElementHandle<T>) -> Memory<T> {
        let Some(policy) = &mut self.actor else {
            todo!()
        };
        let state = handle.lock().unwrap().clone();
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
        *handle.lock().unwrap() = next_state.clone();
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

struct VecWorker<T: R2lTensor, E: Env<Tensor = T>> {
    worker: Worker<T, E>,
    handle: ElementHandle<T>,
}

impl<T: R2lTensor, E: Env<Tensor = T>> VecWorker<T, E> {
    fn new(env: E, handle: ElementHandle<T>) -> Self {
        Self {
            worker: Worker::new(env),
            handle,
        }
    }

    fn step(&mut self) -> Memory<T> {
        self.worker.step(&mut self.handle)
    }

    fn set_policy(&mut self, policy: Box<dyn Actor<Tensor = T>>) {
        self.worker.actor = Some(policy);
    }
}

pub struct VecWorkers<T: R2lTensor, E: Env<Tensor = T>> {
    workers: Vec<VecWorker<T, E>>,
}

impl<T: R2lTensor, E: Env<Tensor = T>> VecWorkers<T, E> {
    pub fn new(workers: Vec<(E, ElementHandle<T>)>) -> Self {
        let workers = workers
            .into_iter()
            .map(|(env, handle)| VecWorker::new(env, handle))
            .collect();
        Self { workers }
    }

    fn step(&mut self) -> MultiMemory<T> {
        let mut multi_memory = MultiMemory::with_capacity(self.workers.len());
        for worker in &mut self.workers {
            multi_memory.push_memory(worker.step());
        }
        multi_memory
    }

    fn step_indexed(&mut self, indices: &[usize]) -> MultiMemory<T> {
        let mut multi_memory = MultiMemory::with_capacity(indices.len());
        for idx in indices {
            multi_memory.push_memory(self.workers[*idx].step());
        }
        multi_memory
    }

    fn set_policy<A: Actor<Tensor = T> + Clone>(&mut self, policy: A) {
        for worker in &mut self.workers {
            worker.set_policy(Box::new(policy.clone()));
        }
    }
}

pub struct ThreadWorker<T: R2lTensor, E: Env<Tensor = T>> {
    worker: Worker<T, E>,
    rx: Receiver<WorkerCommand<T>>,
    tx: Sender<WorkerResult<T>>,
}

impl<T: R2lTensor, E: Env<Tensor = T>> ThreadWorker<T, E> {
    fn new(env: E, rx: Receiver<WorkerCommand<T>>, tx: Sender<WorkerResult<T>>) -> Self {
        Self {
            worker: Worker::new(env),
            rx,
            tx,
        }
    }
}

impl<T: R2lTensor, E: Env<Tensor = T>> ElementWorker for ThreadWorker<T, E> {
    type T = T;

    fn build(&mut self) -> Self::T {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        self.worker.env.reset(seed).unwrap()
    }

    fn work(&mut self, mut handle: ElementHandle<Self::T>) {
        while let Ok(command) = self.rx.recv() {
            match command {
                WorkerCommand::Step => {
                    let memory = self.worker.step(&mut handle);
                    self.tx.send(WorkerResult::Stepped(memory)).unwrap();
                }
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.actor = Some(policy);
                    self.tx.send(WorkerResult::PolicySet).unwrap();
                }
                WorkerCommand::Stop => {
                    self.tx.send(WorkerResult::Stopped).unwrap();
                    break;
                }
            }
        }
    }
}

pub struct ThreadWorkerFactory<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> {
    rx: Receiver<WorkerCommand<T>>,
    tx: Sender<WorkerResult<T>>,
    env_builder: EB,
}

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> ThreadWorkerFactory<T, EB> {
    pub fn new(
        rx: Receiver<WorkerCommand<T>>,
        tx: Sender<WorkerResult<T>>,
        env_builder: EB,
    ) -> Self {
        Self {
            rx,
            tx,
            env_builder,
        }
    }
}

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> ElementWorkerFactory
    for ThreadWorkerFactory<T, EB>
{
    type Worker = ThreadWorker<T, <EB as EnvBuilder>::Env>;

    fn build(self) -> Self::Worker {
        let env = self.env_builder.build_env().unwrap();
        ThreadWorker::new(env, self.rx, self.tx)
    }
}

pub struct ThreadHandle<T: R2lTensor> {
    command_tx: Sender<WorkerCommand<T>>,
    result_rx: Receiver<WorkerResult<T>>,
}

impl<T: R2lTensor> ThreadHandle<T> {
    pub fn new(command_tx: Sender<WorkerCommand<T>>, result_rx: Receiver<WorkerResult<T>>) -> Self {
        Self {
            command_tx,
            result_rx,
        }
    }

    fn send(&self, command: WorkerCommand<T>) {
        self.command_tx.send(command).unwrap();
    }

    fn recv(&self) -> WorkerResult<T> {
        self.result_rx.recv().unwrap()
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
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::Step);
        }
        let mut multi_memory = MultiMemory::with_capacity(self.worker_handles.len());
        for worker_handle in &self.worker_handles {
            let WorkerResult::Stepped(memory) = worker_handle.recv() else {
                unreachable!()
            };
            multi_memory.push_memory(memory);
        }
        multi_memory
    }

    fn step_indexed(&self, indices: &[usize]) -> MultiMemory<T> {
        for idx in indices {
            self.worker_handles[*idx].send(WorkerCommand::Step);
        }
        let mut multi_memory = MultiMemory::with_capacity(indices.len());
        for idx in indices {
            let WorkerResult::Stepped(memory) = self.worker_handles[*idx].recv() else {
                unreachable!()
            };
            multi_memory.push_memory(memory);
        }
        multi_memory
    }

    fn set_policy<A: Actor<Tensor = T> + Clone>(&self, policy: A) {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::SetPolicy(Box::new(policy.clone())));
        }
        for worker_handle in &self.worker_handles {
            let WorkerResult::PolicySet = worker_handle.recv() else {
                unreachable!()
            };
        }
    }

    fn shutdown(&self) {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::Stop);
        }
        for worker_handle in &self.worker_handles {
            let WorkerResult::Stopped = worker_handle.recv() else {
                unreachable!()
            };
        }
    }
}

pub enum WorkerPool<E: Env<Tensor: R2lTensor>> {
    Vec(VecWorkers<E::Tensor, E>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env<Tensor: R2lTensor>> WorkerPool<E> {
    pub fn step_indexed(&mut self, indices: &[usize]) -> MultiMemory<E::Tensor> {
        match self {
            Self::Vec(workers) => workers.step_indexed(indices),
            Self::Thread(workers) => workers.step_indexed(indices),
        }
    }

    pub fn step(&mut self) -> MultiMemory<E::Tensor> {
        match self {
            Self::Vec(workers) => workers.step(),
            Self::Thread(workers) => workers.step(),
        }
    }

    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        match self {
            Self::Vec(workers) => workers.set_policy(policy),
            Self::Thread(workers) => workers.set_policy(policy),
        }
    }

    pub fn shutdown(&mut self) {
        match self {
            Self::Vec(_) => {}
            Self::Thread(threads) => threads.shutdown(),
        }
    }
}
