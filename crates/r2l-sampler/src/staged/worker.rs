use bimodal_array::{ElementHandle, ElementWorker, ElementWorkerFactory};
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, MultiMemory},
    env::{Env, EnvBuilder, Snapshot},
    error::{Error, Result},
    models::Actor,
    rng::{sample_u64, set_seed},
    tensor::R2lTensor,
};

fn worker_interrupted(details: impl Into<String>) -> Error {
    Error::ResourceInterrupted(r2l_core::error::ResourceInterrupted {
        resource: "staged sampler worker".into(),
        details: details.into(),
    })
}

pub enum WorkerCommand<T: R2lTensor> {
    Step,
    SetPolicy(Box<dyn Actor<Tensor = T>>),
    ResetEnv(u64),
    Stop,
}

pub enum WorkerResult<T: R2lTensor> {
    Stepped(Result<Memory<T>>),
    PolicySet,
    EnvReset(Result<()>),
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

    fn step(&mut self, handle: &mut ElementHandle<T>) -> Result<Memory<T>> {
        let Some(policy) = &mut self.actor else {
            return Err(Error::InvalidState {
                operation: "step staged sampler worker".into(),
                details: "no policy has been installed".into(),
            });
        };
        let state = handle.lock().unwrap().clone();
        let action = policy.action(state.clone())?;
        let Snapshot {
            state: next_state,
            reward,
            terminated,
            truncated,
        } = self.env.step(action.clone())?;
        let done = terminated || truncated;
        *handle.lock().unwrap() = if done {
            self.env.reset(sample_u64())?
        } else {
            next_state.clone()
        };
        Ok(Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        })
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

    fn step(&mut self) -> Result<Memory<T>> {
        self.worker.step(&mut self.handle)
    }

    fn set_policy(&mut self, policy: Box<dyn Actor<Tensor = T>>) {
        self.worker.actor = Some(policy);
    }

    fn reset(&mut self) -> Result<()> {
        let state = self.worker.env.reset(sample_u64())?;
        *self.handle.lock().unwrap() = state;
        Ok(())
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

    fn step(&mut self) -> Result<MultiMemory<T>> {
        let mut multi_memory = MultiMemory::with_capacity(self.workers.len());
        for worker in &mut self.workers {
            multi_memory.push_memory(worker.step()?);
        }
        Ok(multi_memory)
    }

    fn step_indexed(&mut self, indices: &[usize]) -> Result<MultiMemory<T>> {
        let mut multi_memory = MultiMemory::with_capacity(indices.len());
        for idx in indices {
            multi_memory.push_memory(self.workers[*idx].step()?);
        }
        Ok(multi_memory)
    }

    fn set_policy<A: Actor<Tensor = T> + Clone>(&mut self, policy: &A) {
        for worker in &mut self.workers {
            worker.set_policy(Box::new(policy.clone()));
        }
    }

    fn reset_all(&mut self) -> Result<()> {
        for worker in &mut self.workers {
            worker.reset()?;
        }
        Ok(())
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
        self.worker.env.reset(sample_u64()).unwrap()
    }

    fn work(&mut self, mut handle: ElementHandle<Self::T>) {
        while let Ok(command) = self.rx.recv() {
            match command {
                WorkerCommand::Step => {
                    let result = self.worker.step(&mut handle);
                    if self.tx.send(WorkerResult::Stepped(result)).is_err() {
                        break;
                    }
                }
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.actor = Some(policy);
                    if self.tx.send(WorkerResult::PolicySet).is_err() {
                        break;
                    }
                }
                WorkerCommand::ResetEnv(seed) => {
                    let result = self.worker.env.reset(seed).map(|state| {
                        *handle.lock().unwrap() = state;
                    });
                    if self.tx.send(WorkerResult::EnvReset(result)).is_err() {
                        break;
                    }
                }
                WorkerCommand::Stop => {
                    let _ = self.tx.send(WorkerResult::Stopped);
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
    worker_seed: u64,
}

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> ThreadWorkerFactory<T, EB> {
    pub fn new(
        rx: Receiver<WorkerCommand<T>>,
        tx: Sender<WorkerResult<T>>,
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

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> ElementWorkerFactory
    for ThreadWorkerFactory<T, EB>
{
    type Worker = ThreadWorker<T, <EB as EnvBuilder>::Env>;

    fn build(self) -> Self::Worker {
        set_seed(self.worker_seed);
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

    fn send(&self, command: WorkerCommand<T>) -> bool {
        self.command_tx.send(command).is_ok()
    }

    fn recv(&self) -> Option<WorkerResult<T>> {
        self.result_rx.recv().ok()
    }
}

pub struct ThreadWorkers<T: R2lTensor> {
    worker_handles: Vec<ThreadHandle<T>>,
}

impl<T: R2lTensor> ThreadWorkers<T> {
    pub fn new(worker_handles: Vec<ThreadHandle<T>>) -> Self {
        Self { worker_handles }
    }

    fn step(&self) -> Result<MultiMemory<T>> {
        for worker_handle in &self.worker_handles {
            let _ = worker_handle.send(WorkerCommand::Step);
        }
        let mut multi_memory = MultiMemory::with_capacity(self.worker_handles.len());
        let mut error = None;
        for worker_handle in &self.worker_handles {
            let result = match worker_handle.recv() {
                Some(WorkerResult::Stepped(result)) => result,
                Some(_) => return Err(worker_interrupted("received an unexpected response")),
                None => return Err(worker_interrupted("result channel disconnected")),
            };
            match result {
                Ok(memory) => multi_memory.push_memory(memory),
                Err(worker_error) => {
                    error.get_or_insert(worker_error);
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(multi_memory),
        }
    }

    fn step_indexed(&self, indices: &[usize]) -> Result<MultiMemory<T>> {
        for idx in indices {
            let _ = self.worker_handles[*idx].send(WorkerCommand::Step);
        }
        let mut multi_memory = MultiMemory::with_capacity(indices.len());
        let mut error = None;
        for idx in indices {
            let result = match self.worker_handles[*idx].recv() {
                Some(WorkerResult::Stepped(result)) => result,
                Some(_) => return Err(worker_interrupted("received an unexpected response")),
                None => return Err(worker_interrupted("result channel disconnected")),
            };
            match result {
                Ok(memory) => multi_memory.push_memory(memory),
                Err(worker_error) => {
                    error.get_or_insert(worker_error);
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(multi_memory),
        }
    }

    fn set_policy<A: Actor<Tensor = T> + Clone>(&self, policy: &A) -> Result<()> {
        for worker_handle in &self.worker_handles {
            if !worker_handle.send(WorkerCommand::SetPolicy(Box::new(policy.clone()))) {
                return Err(worker_interrupted("command channel disconnected"));
            }
        }
        for worker_handle in &self.worker_handles {
            match worker_handle.recv() {
                Some(WorkerResult::PolicySet) => {}
                Some(_) => return Err(worker_interrupted("received an unexpected response")),
                None => return Err(worker_interrupted("result channel disconnected")),
            }
        }
        Ok(())
    }

    fn reset_all(&self) -> Result<()> {
        for worker_handle in &self.worker_handles {
            let _ = worker_handle.send(WorkerCommand::ResetEnv(sample_u64()));
        }
        let mut result = Ok(());
        for worker_handle in &self.worker_handles {
            let worker_result = match worker_handle.recv() {
                Some(WorkerResult::EnvReset(worker_result)) => worker_result,
                Some(_) => return Err(worker_interrupted("received an unexpected response")),
                None => return Err(worker_interrupted("result channel disconnected")),
            };
            if result.is_ok() {
                result = worker_result;
            }
        }
        result
    }

    fn shutdown(&mut self) {
        let worker_handles = std::mem::take(&mut self.worker_handles);
        for worker_handle in &worker_handles {
            let _ = worker_handle.send(WorkerCommand::Stop);
        }
        for worker_handle in worker_handles {
            let _ = worker_handle.recv();
        }
    }
}

pub enum WorkerPool<E: Env<Tensor: R2lTensor>> {
    Vec(VecWorkers<E::Tensor, E>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env<Tensor: R2lTensor>> WorkerPool<E> {
    pub fn step_indexed(&mut self, indices: &[usize]) -> Result<MultiMemory<E::Tensor>> {
        match self {
            Self::Vec(workers) => workers.step_indexed(indices),
            Self::Thread(workers) => workers.step_indexed(indices),
        }
    }

    pub fn step(&mut self) -> Result<MultiMemory<E::Tensor>> {
        match self {
            Self::Vec(workers) => workers.step(),
            Self::Thread(workers) => workers.step(),
        }
    }

    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: &A) -> Result<()> {
        match self {
            Self::Vec(workers) => {
                workers.set_policy(policy);
                Ok(())
            }
            Self::Thread(workers) => workers.set_policy(policy),
        }
    }

    pub fn reset_all(&mut self) -> Result<()> {
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

impl<E: Env<Tensor: R2lTensor>> Drop for WorkerPool<E> {
    fn drop(&mut self) {
        self.shutdown();
    }
}
