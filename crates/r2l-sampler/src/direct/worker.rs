use std::thread::JoinHandle;

use bimodal_array::ElementHandle;
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, buffer::TrajectoryBuffer},
    env::{Env, EnvDescription, Snapshot},
    models::Actor,
    rng::{env_seed, env_worker_rng},
    tensor::R2lTensor,
};
use rand::{RngExt, rngs::StdRng};

use crate::direct::RolloutMode;

pub(crate) type CommandSender<T> = Sender<WorkerCommand<T>>;
pub(crate) type CommandReceiver<T> = Receiver<WorkerCommand<T>>;

pub(crate) type ResultSender<T> = Sender<WorkerResult<T>>;
pub(crate) type ResultReceiver<T> = Receiver<WorkerResult<T>>;

pub fn step_env<T: R2lTensor, E: Env<Tensor = T>>(
    env: &mut E,
    actor: &mut Box<dyn Actor<Tensor = T>>,
    last_state: Option<T>,
    rng: &mut StdRng,
) -> Memory<T> {
    let state = if let Some(state) = last_state {
        state
    } else {
        env.reset(rng.random::<u64>()).unwrap()
    };
    let action = actor.action(state.clone()).unwrap();
    let Snapshot {
        state: mut next_state,
        reward,
        terminated,
        truncated,
    } = env.step(action.clone()).unwrap();
    let done = terminated || truncated;
    if done {
        next_state = env.reset(rng.random::<u64>()).unwrap();
    }
    Memory {
        state,
        next_state,
        action,
        reward,
        terminated,
        truncated,
    }
}

pub enum WorkerCommand<T: R2lTensor> {
    SetPolicy(Box<dyn Actor<Tensor = T>>),
    Collect(RolloutMode),
    ResetEnv(u64),
    ClearBuffer,
    GetEnvDescription,
    Shutdown,
    GetLastState,
    SetLastState(T),
    ResetEnvUninserted(u64),
    ReplaceLastNextState(T),
}

pub enum WorkerResult<T: R2lTensor> {
    PolicySet,
    Collected,
    EnvReset,
    BufferCleared,
    EnvDescription(EnvDescription<T>),
    Shutdown,
    LastState(Option<T>),
    LastStateSet,
    ResetEnvUninsertedResult(T),
    LastNextStateReplaced,
}

pub struct ThreadHandle<T: R2lTensor> {
    handle: JoinHandle<()>,
    command_tx: CommandSender<T>,
    worker_rx: ResultReceiver<T>,
}

impl<T: R2lTensor> ThreadHandle<T> {
    pub fn new(
        handle: JoinHandle<()>,
        command_tx: CommandSender<T>,
        worker_rx: ResultReceiver<T>,
    ) -> Self {
        Self {
            handle,
            command_tx,
            worker_rx,
        }
    }

    pub fn env_description(&self) -> EnvDescription<T> {
        self.command_tx
            .send(WorkerCommand::GetEnvDescription)
            .unwrap();
        let WorkerResult::EnvDescription(env_description) = self.worker_rx.recv().unwrap() else {
            todo!()
        };
        env_description
    }

    pub fn send(&self, command: WorkerCommand<T>) {
        self.command_tx.send(command).unwrap();
    }

    pub fn recv(&self) -> WorkerResult<T> {
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
    env_rng: StdRng,
}

impl<E: Env> Worker<E> {
    pub fn new(
        env: E,
        buffer: ElementHandle<TrajectoryBuffer<E::Tensor>>,
        worker_idx: usize,
    ) -> Self {
        Self {
            env,
            buffer,
            actor: None,
            last_state: None,
            env_rng: env_worker_rng(worker_idx),
        }
    }

    pub fn set_last_state(&mut self, last_state: E::Tensor) {
        self.last_state = Some(last_state);
    }

    pub fn replace_last_next_state(&mut self, next_state: E::Tensor) {
        self.buffer
            .lock()
            .unwrap()
            .replace_last_next_state(next_state);
    }

    pub fn clear(&mut self) {
        self.buffer.lock().unwrap().clear();
    }

    pub fn collect(&mut self, bound: RolloutMode) {
        let Some(actor) = &mut self.actor else {
            todo!()
        };
        let mut buffer = self.buffer.lock().unwrap();
        match bound {
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = 0;
                loop {
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, actor, last_state, &mut self.env_rng);
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
                    let memory = step_env(&mut self.env, actor, last_state, &mut self.env_rng);
                    self.last_state = Some(memory.next_state.clone());
                    buffer.push(memory);
                }
            }
        }
    }

    // resets the initial state and clears the buffer. Used by the Evaluator hook
    pub fn reset(&mut self, seed: u64) {
        let state = self.env.reset(seed).unwrap();
        self.last_state = Some(state);
        self.buffer.lock().unwrap().clear();
    }

    pub fn reset_env_uninserted(&mut self, seed: u64) -> E::Tensor {
        self.env.reset(seed).unwrap()
    }
}

pub struct ThreadWorker<E: Env> {
    worker: Worker<E>,
    rx: CommandReceiver<E::Tensor>,
    tx: ResultSender<E::Tensor>,
}

impl<E: Env> ThreadWorker<E> {
    pub fn new(
        worker: Worker<E>,
        rx: CommandReceiver<E::Tensor>,
        tx: ResultSender<E::Tensor>,
    ) -> Self {
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
                    self.worker.collect(bound);
                    self.tx.send(WorkerResult::Collected).unwrap();
                }
                WorkerCommand::GetEnvDescription => {
                    let environment_descriotion = self.worker.env.env_description();
                    self.tx
                        .send(WorkerResult::EnvDescription(environment_descriotion))
                        .unwrap();
                }
                WorkerCommand::Shutdown => {
                    self.tx.send(WorkerResult::Shutdown).unwrap();
                    break;
                }
                WorkerCommand::ResetEnv(seed) => {
                    self.worker.reset(seed);
                    self.tx.send(WorkerResult::EnvReset).unwrap();
                }
                WorkerCommand::ClearBuffer => {
                    self.worker.clear();
                    self.tx.send(WorkerResult::BufferCleared).unwrap();
                }
                WorkerCommand::GetLastState => {
                    let last_state = self.worker.last_state.clone();
                    self.tx.send(WorkerResult::LastState(last_state)).unwrap();
                }
                WorkerCommand::SetLastState(state) => {
                    self.worker.set_last_state(state);
                    self.tx.send(WorkerResult::LastStateSet).unwrap();
                }
                WorkerCommand::ResetEnvUninserted(seed) => {
                    let state = self.worker.reset_env_uninserted(seed);
                    self.tx
                        .send(WorkerResult::ResetEnvUninsertedResult(state))
                        .unwrap();
                }
                WorkerCommand::ReplaceLastNextState(state) => {
                    self.worker.replace_last_next_state(state);
                    self.tx.send(WorkerResult::LastNextStateReplaced).unwrap();
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

    // TODO: this can fail. We need to mark this as failible once we figured the right Error types out
    pub fn env_description(&self) -> EnvDescription<T> {
        self.worker_handles[0].env_description()
    }

    pub fn set_policy<A: Actor<Tensor = T> + Clone>(&self, policy: A) {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::SetPolicy(Box::new(policy.clone())));
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }

    pub fn collect_rollout(&self, bound: RolloutMode) {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::Collect(bound));
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }

    pub fn reset_all(&self) {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::ResetEnv(env_seed()));
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }

    pub fn get_last_states(&self) -> Option<Vec<T>> {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::GetLastState);
        }
        self.worker_handles
            .iter()
            .map(|h| {
                let WorkerResult::LastState(last_state) = h.recv() else {
                    unreachable!()
                };
                last_state
            })
            .collect()
    }

    pub fn set_last_states(&self, states: Vec<T>) {
        for (worker_handle, state) in self.worker_handles.iter().zip(states) {
            worker_handle.send(WorkerCommand::SetLastState(state));
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }

    pub fn reset_envs_uninserted(&self) -> Vec<T> {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::ResetEnvUninserted(env_seed()));
        }
        self.worker_handles
            .iter()
            .map(|wh| {
                let WorkerResult::ResetEnvUninsertedResult(state) = wh.recv() else {
                    unreachable!()
                };
                state
            })
            .collect()
    }

    pub fn replace_last_next_states(&self, states: Vec<T>) {
        for (worker_handle, state) in self.worker_handles.iter().zip(states) {
            worker_handle.send(WorkerCommand::ReplaceLastNextState(state));
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }

    pub fn shutdown(&mut self) {
        // shutdown one by one.
        while let Some(worker) = self.worker_handles.pop() {
            worker.shutdown();
        }
    }

    pub fn clear_buffers(&mut self) {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::ClearBuffer);
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }
}

pub enum WorkerPool<E: Env> {
    Vec(Vec<Worker<E>>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env> WorkerPool<E> {
    pub fn clear_buffers(&mut self) {
        match self {
            Self::Vec(workers) => {
                workers.iter_mut().for_each(|w| w.clear());
            }
            Self::Thread(thread) => {
                thread.clear_buffers();
            }
        }
    }

    pub fn env_description(&self) -> EnvDescription<E::Tensor> {
        match self {
            Self::Vec(workers) => workers[0].env.env_description(),
            Self::Thread(tw) => tw.env_description(),
        }
    }

    pub fn set_actor<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        match self {
            Self::Vec(workers) => {
                for worker in workers.iter_mut() {
                    worker.actor = Some(Box::new(policy.clone()))
                }
            }
            Self::Thread(thread_workers) => {
                thread_workers.set_policy(policy);
            }
        }
    }

    pub fn collect(&mut self, bound: RolloutMode) {
        match self {
            Self::Vec(workers) => {
                for worker in workers {
                    worker.collect(bound);
                }
            }
            Self::Thread(thread_workers) => {
                thread_workers.collect_rollout(bound);
            }
        }
    }

    pub fn single_step(&mut self) {
        self.collect(RolloutMode::StepBound { n_steps: 1 });
    }

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

    pub fn reset_all_envs(&mut self) {
        match self {
            Self::Vec(workers) => {
                for worker in workers {
                    worker.reset(env_seed());
                }
            }
            Self::Thread(workers) => {
                workers.reset_all();
            }
        }
    }

    pub fn get_last_states(&mut self) -> Option<Vec<E::Tensor>> {
        match self {
            Self::Vec(workers) => {
                // in the order of the workers
                workers.iter().map(|w| w.last_state.clone()).collect()
            }
            Self::Thread(workers) => {
                // worker pools ensures the order
                workers.get_last_states()
            }
        }
    }

    pub fn set_last_states(&mut self, states: Vec<E::Tensor>) {
        match self {
            Self::Vec(workers) => {
                for (worker, state) in workers.iter_mut().zip(states) {
                    worker.set_last_state(state)
                }
            }
            Self::Thread(workers) => {
                workers.set_last_states(states);
            }
        }
    }

    pub fn replace_last_next_states(&mut self, states: Vec<E::Tensor>) {
        match self {
            Self::Vec(workers) => {
                for (worker, state) in workers.iter_mut().zip(states) {
                    worker.replace_last_next_state(state);
                }
            }
            Self::Thread(workers) => workers.replace_last_next_states(states),
        }
    }

    pub fn reset_envs_uninserted(&mut self) -> Vec<E::Tensor> {
        match self {
            Self::Vec(workers) => {
                // resets all the envs but does not set it as a last state
                workers
                    .iter_mut()
                    .map(|w| w.reset_env_uninserted(env_seed()))
                    .collect()
            }
            Self::Thread(workers) => workers.reset_envs_uninserted(),
        }
    }
}

impl<E: Env> Drop for WorkerPool<E> {
    fn drop(&mut self) {
        self.shutdown();
    }
}
