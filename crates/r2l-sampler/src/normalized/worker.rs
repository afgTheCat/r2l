use bimodal_array::{ElementHandle, ElementWorker, ElementWorkerFactory};
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, MultiMemory},
    env::{Env, EnvBuilder, Snapshot},
    models::Actor,
    rng::{sample_u64, set_seed},
    tensor::R2lTensor,
};

pub enum WorkerCommand<T: R2lTensor, S: Clone + Send + Sync + 'static> {
    Step,
    SetPolicy(Box<dyn Actor<Tensor = T, State = S>>),
    ResetEnv(u64),
    Stop,
}

pub enum WorkerResult<T: R2lTensor, S: Clone + Send + Sync + 'static> {
    Stepped(Memory<T, S>),
    PolicySet,
    EnvReset,
    Stopped,
}

struct Worker<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static> {
    actor: Option<Box<dyn Actor<Tensor = E::Tensor, State = S>>>,
    actor_state: Option<S>,
    env: E,
}

impl<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static> Worker<T, E, S> {
    fn new(env: E) -> Self {
        Self {
            actor: None,
            actor_state: None,
            env,
        }
    }

    fn set_actor(&mut self, actor: Box<dyn Actor<Tensor = E::Tensor, State = S>>) {
        self.actor = Some(actor);
    }

    fn step(&mut self, handle: &mut ElementHandle<T>) -> Memory<T, S> {
        let Some(policy) = &self.actor else { todo!() };
        let state = handle.lock().unwrap().clone();
        let actor_state = self.actor_state.take();
        let memory_actor_state = actor_state.clone();
        let (action, next_actor_state) = policy.action(state.clone(), actor_state).unwrap();
        let Snapshot {
            state: mut next_state,
            reward,
            terminated,
            truncated,
        } = self.env.step(action.clone()).unwrap();
        let done = terminated || truncated;
        if done {
            next_state = self.env.reset(sample_u64()).unwrap();
        }
        self.actor_state = (!done).then_some(next_actor_state);
        *handle.lock().unwrap() = next_state.clone();
        Memory {
            state,
            next_state,
            action,
            actor_state: memory_actor_state,
            reward,
            terminated,
            truncated,
        }
    }
}

struct VecWorker<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static> {
    worker: Worker<T, E, S>,
    handle: ElementHandle<T>,
}

impl<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static> VecWorker<T, E, S> {
    fn new(env: E, handle: ElementHandle<T>) -> Self {
        Self {
            worker: Worker::new(env),
            handle,
        }
    }

    fn step(&mut self) -> Memory<T, S> {
        self.worker.step(&mut self.handle)
    }

    fn set_policy(&mut self, policy: Box<dyn Actor<Tensor = T, State = S>>) {
        self.worker.set_actor(policy);
    }

    fn reset(&mut self) {
        let state = self.worker.env.reset(sample_u64()).unwrap();
        self.worker.actor_state = None;
        *self.handle.lock().unwrap() = state;
    }
}

pub struct VecWorkers<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static = ()> {
    workers: Vec<VecWorker<T, E, S>>,
}

impl<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static> VecWorkers<T, E, S> {
    pub fn new(workers: Vec<(E, ElementHandle<T>)>) -> Self {
        let workers = workers
            .into_iter()
            .map(|(env, handle)| VecWorker::new(env, handle))
            .collect();
        Self { workers }
    }

    fn step(&mut self) -> MultiMemory<T, S> {
        let mut multi_memory = MultiMemory::with_capacity(self.workers.len());
        for worker in &mut self.workers {
            multi_memory.push_memory(worker.step());
        }
        multi_memory
    }

    fn step_indexed(&mut self, indices: &[usize]) -> MultiMemory<T, S> {
        let mut multi_memory = MultiMemory::with_capacity(indices.len());
        for idx in indices {
            multi_memory.push_memory(self.workers[*idx].step());
        }
        multi_memory
    }

    fn set_policy<A: Actor<Tensor = T, State = S> + Clone>(&mut self, policy: A) {
        for worker in &mut self.workers {
            worker.set_policy(Box::new(policy.clone()));
        }
    }

    fn reset_all(&mut self) {
        for worker in &mut self.workers {
            worker.reset();
        }
    }
}

pub struct ThreadWorker<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static = ()> {
    worker: Worker<T, E, S>,
    rx: Receiver<WorkerCommand<T, S>>,
    tx: Sender<WorkerResult<T, S>>,
}

impl<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static> ThreadWorker<T, E, S> {
    fn new(env: E, rx: Receiver<WorkerCommand<T, S>>, tx: Sender<WorkerResult<T, S>>) -> Self {
        Self {
            worker: Worker::new(env),
            rx,
            tx,
        }
    }
}

impl<T: R2lTensor, E: Env<Tensor = T>, S: Clone + Send + Sync + 'static> ElementWorker
    for ThreadWorker<T, E, S>
{
    type T = T;

    fn build(&mut self) -> Self::T {
        self.worker.env.reset(sample_u64()).unwrap()
    }

    fn work(&mut self, mut handle: ElementHandle<Self::T>) {
        while let Ok(command) = self.rx.recv() {
            match command {
                WorkerCommand::Step => {
                    let memory = self.worker.step(&mut handle);
                    self.tx.send(WorkerResult::Stepped(memory)).unwrap();
                }
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.set_actor(policy);
                    self.tx.send(WorkerResult::PolicySet).unwrap();
                }
                WorkerCommand::ResetEnv(seed) => {
                    let state = self.worker.env.reset(seed).unwrap();
                    self.worker.actor_state = None;
                    *handle.lock().unwrap() = state;
                    self.tx.send(WorkerResult::EnvReset).unwrap();
                }
                WorkerCommand::Stop => {
                    self.tx.send(WorkerResult::Stopped).unwrap();
                    break;
                }
            }
        }
    }
}

pub struct ThreadWorkerFactory<
    T: R2lTensor,
    EB: EnvBuilder<Env: Env<Tensor = T>>,
    S: Clone + Send + Sync + 'static = (),
> {
    rx: Receiver<WorkerCommand<T, S>>,
    tx: Sender<WorkerResult<T, S>>,
    env_builder: EB,
    worker_seed: u64,
}

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>, S: Clone + Send + Sync + 'static>
    ThreadWorkerFactory<T, EB, S>
{
    pub fn new(
        rx: Receiver<WorkerCommand<T, S>>,
        tx: Sender<WorkerResult<T, S>>,
        env_builder: EB,
        worker_seed: u64,
    ) -> Self {
        Self {
            rx,
            tx,
            env_builder,
            worker_seed,
        }
    }
}

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>, S: Clone + Send + Sync + 'static>
    ElementWorkerFactory for ThreadWorkerFactory<T, EB, S>
{
    type Worker = ThreadWorker<T, <EB as EnvBuilder>::Env, S>;

    fn build(self) -> Self::Worker {
        set_seed(self.worker_seed);
        let env = self.env_builder.build_env().unwrap();
        ThreadWorker::new(env, self.rx, self.tx)
    }
}

pub struct ThreadHandle<T: R2lTensor, S: Clone + Send + Sync + 'static = ()> {
    command_tx: Sender<WorkerCommand<T, S>>,
    result_rx: Receiver<WorkerResult<T, S>>,
}

impl<T: R2lTensor, S: Clone + Send + Sync + 'static> ThreadHandle<T, S> {
    pub fn new(
        command_tx: Sender<WorkerCommand<T, S>>,
        result_rx: Receiver<WorkerResult<T, S>>,
    ) -> Self {
        Self {
            command_tx,
            result_rx,
        }
    }

    fn send(&self, command: WorkerCommand<T, S>) {
        self.command_tx.send(command).unwrap();
    }

    fn recv(&self) -> WorkerResult<T, S> {
        self.result_rx.recv().unwrap()
    }
}

pub struct ThreadWorkers<T: R2lTensor, S: Clone + Send + Sync + 'static = ()> {
    worker_handles: Vec<ThreadHandle<T, S>>,
}

impl<T: R2lTensor, S: Clone + Send + Sync + 'static> ThreadWorkers<T, S> {
    pub fn new(worker_handles: Vec<ThreadHandle<T, S>>) -> Self {
        Self { worker_handles }
    }

    fn step(&self) -> MultiMemory<T, S> {
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

    fn step_indexed(&self, indices: &[usize]) -> MultiMemory<T, S> {
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

    fn set_policy<A: Actor<Tensor = T, State = S> + Clone>(&self, policy: A) {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::SetPolicy(Box::new(policy.clone())));
        }
        for worker_handle in &self.worker_handles {
            let WorkerResult::PolicySet = worker_handle.recv() else {
                unreachable!()
            };
        }
    }

    fn reset_all(&self) {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::ResetEnv(sample_u64()));
        }
        for worker_handle in &self.worker_handles {
            let WorkerResult::EnvReset = worker_handle.recv() else {
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

pub enum WorkerPool<E: Env<Tensor: R2lTensor>, S: Clone + Send + Sync + 'static = ()> {
    Vec(VecWorkers<E::Tensor, E, S>),
    Thread(ThreadWorkers<E::Tensor, S>),
}

impl<E: Env<Tensor: R2lTensor>, S: Clone + Send + Sync + 'static> WorkerPool<E, S> {
    pub fn step_indexed(&mut self, indices: &[usize]) -> MultiMemory<E::Tensor, S> {
        match self {
            Self::Vec(workers) => workers.step_indexed(indices),
            Self::Thread(workers) => workers.step_indexed(indices),
        }
    }

    pub fn step(&mut self) -> MultiMemory<E::Tensor, S> {
        match self {
            Self::Vec(workers) => workers.step(),
            Self::Thread(workers) => workers.step(),
        }
    }

    pub fn set_policy<A: Actor<Tensor = E::Tensor, State = S> + Clone>(&mut self, policy: A) {
        match self {
            Self::Vec(workers) => workers.set_policy(policy),
            Self::Thread(workers) => workers.set_policy(policy),
        }
    }

    pub fn reset_all(&mut self) {
        match self {
            Self::Vec(workers) => workers.reset_all(),
            Self::Thread(workers) => workers.reset_all(),
        }
    }

    pub fn shutdown(&mut self) {
        match self {
            Self::Vec(_) => {}
            Self::Thread(threads) => threads.shutdown(),
        }
    }
}
