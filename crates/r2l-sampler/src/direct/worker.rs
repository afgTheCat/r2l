use std::thread::JoinHandle;

use bimodal_array::ElementHandle;
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, buffer::TrajectoryBuffer},
    env::{Env, Snapshot},
    error::Error,
    models::Actor,
    rng::sample_u64,
    tensor::R2lTensor,
};

use crate::direct::RolloutMode;

pub(crate) type CommandSender<T> = Sender<WorkerCommand<T>>;
pub(crate) type CommandReceiver<T> = Receiver<WorkerCommand<T>>;

pub(crate) type ResultSender = Sender<WorkerResult>;
pub(crate) type ResultReceiver = Receiver<WorkerResult>;

pub fn step_env<T: R2lTensor, E: Env<Tensor = T>>(
    env: &mut E,
    actor: &mut Box<dyn Actor<Tensor = T>>,
    last_state: Option<T>,
) -> Result<Memory<T>, Error> {
    let state = if let Some(state) = last_state {
        state
    } else {
        env.reset(sample_u64())?
    };
    let action = actor.action(state.clone()).unwrap();
    let Snapshot {
        state: mut next_state,
        reward,
        terminated,
        truncated,
    } = env.step(action.clone())?;
    let done = terminated || truncated;
    if done {
        next_state = env.reset(sample_u64())?;
    }
    Ok(Memory {
        state,
        next_state,
        action,
        reward,
        terminated,
        truncated,
    })
}

pub enum WorkerCommand<T: R2lTensor> {
    SetPolicy(Box<dyn Actor<Tensor = T>>),
    Collect(RolloutMode),
    ResetEnv(u64),
    ClearBuffer,
    Shutdown,
}

pub enum WorkerResult {
    PolicySet,
    Collected(Result<(), Error>),
    EnvReset(Result<(), Error>),
    BufferCleared,
    Shutdown,
}

pub struct ThreadHandle<T: R2lTensor> {
    handle: JoinHandle<()>,
    command_tx: CommandSender<T>,
    worker_rx: ResultReceiver,
}

impl<T: R2lTensor> ThreadHandle<T> {
    pub fn new(
        handle: JoinHandle<()>,
        command_tx: CommandSender<T>,
        worker_rx: ResultReceiver,
    ) -> Self {
        Self {
            handle,
            command_tx,
            worker_rx,
        }
    }

    pub fn send(&self, command: WorkerCommand<T>) {
        self.command_tx.send(command).unwrap();
    }

    pub fn recv(&self) -> WorkerResult {
        self.worker_rx.recv().unwrap()
    }

    pub fn shutdown(self) {
        self.command_tx.send(WorkerCommand::Shutdown).unwrap();
        self.worker_rx.recv().unwrap();
        self.handle.join().unwrap();
    }
}

pub struct Worker<E: Env> {
    pub env: E,
    pub buffer: ElementHandle<TrajectoryBuffer<E::Tensor>>,
    pub actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    pub last_state: Option<E::Tensor>,
}

impl<E: Env> Worker<E> {
    pub fn new(env: E, buffer: ElementHandle<TrajectoryBuffer<E::Tensor>>) -> Self {
        Self {
            env,
            buffer,
            actor: None,
            last_state: None,
        }
    }

    pub fn clear(&mut self) {
        self.buffer.lock().unwrap().clear();
    }

    pub fn collect(&mut self, bound: RolloutMode) -> Result<(), Error> {
        let Some(actor) = &mut self.actor else {
            unreachable!()
        };
        let mut buffer = self.buffer.lock().unwrap();
        match bound {
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = 0;
                loop {
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, actor, last_state)?;
                    let terminates = memory.is_done();
                    self.last_state = Some(memory.next_state.clone());
                    buffer.push(memory);
                    if terminates {
                        episodes += 1;
                    }
                    if episodes >= n_episodes {
                        break;
                    }
                }
            }
            RolloutMode::StepBound { n_steps } => {
                for _ in 0..n_steps {
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, actor, last_state)?;
                    self.last_state = Some(memory.next_state.clone());
                    buffer.push(memory);
                }
            }
        }
        Ok(())
    }

    // resets the initial state and clears the buffer. Used by the Evaluator hook
    pub fn reset(&mut self, seed: u64) -> Result<(), Error> {
        let state = self.env.reset(seed)?;
        self.last_state = Some(state);
        self.buffer.lock().unwrap().clear();
        Ok(())
    }
}

pub struct ThreadWorker<E: Env> {
    worker: Worker<E>,
    rx: CommandReceiver<E::Tensor>,
    tx: ResultSender,
}

impl<E: Env> ThreadWorker<E> {
    pub fn new(worker: Worker<E>, rx: CommandReceiver<E::Tensor>, tx: ResultSender) -> Self {
        Self { worker, rx, tx }
    }

    pub fn work(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.actor = Some(policy);
                    self.tx.send(WorkerResult::PolicySet).unwrap();
                }
                WorkerCommand::Collect(bound) => {
                    let result = self.worker.collect(bound);
                    self.tx.send(WorkerResult::Collected(result)).unwrap();
                }
                WorkerCommand::Shutdown => {
                    self.tx.send(WorkerResult::Shutdown).unwrap();
                    break;
                }
                WorkerCommand::ResetEnv(seed) => {
                    let result = self.worker.reset(seed);
                    self.tx.send(WorkerResult::EnvReset(result)).unwrap();
                }
                WorkerCommand::ClearBuffer => {
                    self.worker.clear();
                    self.tx.send(WorkerResult::BufferCleared).unwrap();
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

    pub fn set_policy<A: Actor<Tensor = T> + Clone>(&self, policy: &A) {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::SetPolicy(Box::new(policy.clone())));
        }
        for worker_handle in &self.worker_handles {
            worker_handle.recv();
        }
    }

    pub fn collect_rollout(&self, bound: RolloutMode) -> Result<(), Error> {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::Collect(bound));
        }
        let mut result = Ok(());
        for worker_handle in &self.worker_handles {
            let WorkerResult::Collected(worker_result) = worker_handle.recv() else {
                unreachable!()
            };
            if result.is_ok() {
                result = worker_result;
            }
        }
        result
    }

    pub fn reset_all(&self) -> Result<(), Error> {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::ResetEnv(sample_u64()));
        }
        let mut result = Ok(());
        for worker_handle in &self.worker_handles {
            let WorkerResult::EnvReset(worker_result) = worker_handle.recv() else {
                unreachable!()
            };
            if result.is_ok() {
                result = worker_result;
            }
        }
        result
    }

    pub fn shutdown(&mut self) {
        // shutdown one by one.
        while let Some(worker) = self.worker_handles.pop() {
            worker.shutdown();
        }
    }

    pub fn clear_buffers(&mut self) {
        for worker_handle in &self.worker_handles {
            worker_handle.send(WorkerCommand::ClearBuffer);
        }
        for worker_handle in &self.worker_handles {
            worker_handle.recv();
        }
    }
}

/// Pool of inline or threaded environment workers.
pub enum WorkerPool<E: Env> {
    /// Workers stepped sequentially on the calling thread.
    Vec(Vec<Worker<E>>),
    /// Workers stepped on dedicated background threads.
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env> WorkerPool<E> {
    /// Clears every worker's trajectory buffer.
    pub fn clear_buffers(&mut self) {
        match self {
            Self::Vec(workers) => {
                workers.iter_mut().for_each(Worker::clear);
            }
            Self::Thread(thread) => {
                thread.clear_buffers();
            }
        }
    }

    /// Installs a clone of `policy` on every worker.
    pub fn set_actor<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: &A) {
        match self {
            Self::Vec(workers) => {
                for worker in workers.iter_mut() {
                    worker.actor = Some(Box::new(policy.clone()));
                }
            }
            Self::Thread(thread_workers) => {
                thread_workers.set_policy(policy);
            }
        }
    }

    /// Collects one bounded rollout on every worker.
    pub fn collect(&mut self, bound: RolloutMode) -> Result<(), Error> {
        match self {
            Self::Vec(workers) => {
                for worker in workers {
                    worker.collect(bound)?;
                }
                Ok(())
            }
            Self::Thread(thread_workers) => thread_workers.collect_rollout(bound),
        }
    }

    /// Stops and joins threaded workers.
    pub fn shutdown(&mut self) {
        match self {
            Self::Vec(_) => {
                // No need to explicitly shut down
            }
            Self::Thread(workers) => {
                workers.shutdown();
            }
        }
    }

    /// Resets all worker environments with fresh seeds.
    pub fn reset_all_envs(&mut self) -> Result<(), Error> {
        match self {
            Self::Vec(workers) => {
                for worker in workers {
                    worker.reset(sample_u64())?;
                }
                Ok(())
            }
            Self::Thread(workers) => workers.reset_all(),
        }
    }
}

impl<E: Env> Drop for WorkerPool<E> {
    fn drop(&mut self) {
        self.shutdown();
    }
}
