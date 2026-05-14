use std::thread::JoinHandle;

use bimodal_array::ElementHandle;
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{ExpandableTrajectoryContainer, Memory},
    env::{Env, EnvDescription, Snapshot},
    models::Actor,
    rng::RNG,
    tensor::R2lTensor,
};
use rand::RngExt;

use crate::RolloutMode;

type CommandSender<T> = Sender<WorkerCommand<T>>;
type CommandReceiver<T> = Receiver<WorkerCommand<T>>;

type ResultSender<T> = Sender<WorkerResult<T>>;
type ResultReceiver<T> = Receiver<WorkerResult<T>>;

pub struct Worker<E: Env, D: ExpandableTrajectoryContainer<Tensor = E::Tensor>> {
    pub env: E,
    pub buffer: ElementHandle<D>,
    // TODO: this is a bit archaic, we might want somethings else here
    pub policy: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    pub last_state: Option<E::Tensor>,
}

pub fn step_env<T: R2lTensor, E: Env<Tensor = T>>(
    env: &mut E,
    policy: &mut Box<dyn Actor<Tensor = T>>,
    last_state: Option<T>,
) -> Memory<T> {
    let state = if let Some(state) = last_state {
        state
    } else {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        env.reset(seed).unwrap()
    };
    let action = policy.action(state.clone()).unwrap();
    let Snapshot {
        state: mut next_state,
        reward,
        terminated,
        truncated,
    } = env.step(action.clone()).unwrap();
    let done = terminated || truncated;
    if done {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        next_state = env.reset(seed).unwrap();
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

impl<E: Env, D: ExpandableTrajectoryContainer<Tensor = E::Tensor>> Worker<E, D> {
    pub fn new(env: E, buffer: ElementHandle<D>) -> Self {
        Self {
            env,
            buffer,
            policy: None,
            last_state: None,
        }
    }

    pub fn collect_rollout(&mut self, bound: RolloutMode) {
        let Some(policy) = &mut self.policy else {
            todo!()
        };
        let mut buffer = self.buffer.lock().unwrap();
        buffer.begin_rollout();
        match bound {
            RolloutMode::StepBound { n_steps: steps } => {
                for _ in 0..steps {
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, policy, last_state);
                    self.last_state = Some(memory.next_state.clone());
                    buffer.push(memory);
                }
            }
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = 0;
                loop {
                    // TODO: this is a bit awkward.
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, policy, last_state);
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
        }
    }

    pub fn reset(&mut self, seed: u64) {
        let state = self.env.reset(seed).unwrap();
        self.last_state = Some(state);
    }
}

pub enum WorkerCommand<T: R2lTensor> {
    SetPolicy(Box<dyn Actor<Tensor = T>>),
    Collect(RolloutMode),
    ResetEnv(u64),
    GetEnvDescription,
    Shutdown,
}

pub enum WorkerResult<T: R2lTensor> {
    PolicySet,
    Collected,
    EnvReset,
    EnvDescription(EnvDescription<T>),
    Shutdown,
}

pub struct ThreadWorker<E: Env, D: ExpandableTrajectoryContainer<Tensor = E::Tensor>> {
    worker: Worker<E, D>,
    rx: CommandReceiver<E::Tensor>,
    tx: ResultSender<E::Tensor>,
}

impl<E: Env, D: ExpandableTrajectoryContainer<Tensor = E::Tensor>> ThreadWorker<E, D> {
    pub fn new(
        worker: Worker<E, D>,
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
                    self.worker.policy = Some(policy);
                    self.tx.send(WorkerResult::PolicySet).unwrap();
                }
                WorkerCommand::Collect(bound) => {
                    self.worker.collect_rollout(bound);
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
            }
        }
    }
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
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            worker_handle.send(WorkerCommand::ResetEnv(seed));
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
}

pub enum WorkerPool<E: Env, B: ExpandableTrajectoryContainer<Tensor = <E as Env>::Tensor>> {
    Vec(Vec<Worker<E, B>>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env, B: ExpandableTrajectoryContainer<Tensor = <E as Env>::Tensor>> WorkerPool<E, B> {
    pub fn env_description(&self) -> EnvDescription<E::Tensor> {
        match self {
            Self::Vec(workers) => workers[0].env.env_description(),
            Self::Thread(tw) => tw.env_description(),
        }
    }

    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        match self {
            WorkerPool::Vec(workers) => {
                for worker in workers.iter_mut() {
                    worker.policy = Some(Box::new(policy.clone()))
                }
            }
            WorkerPool::Thread(thread_workers) => {
                thread_workers.set_policy(policy);
            }
        }
    }

    pub fn collect(&mut self, bound: RolloutMode) {
        match self {
            WorkerPool::Vec(workers) => {
                for worker in workers {
                    worker.collect_rollout(bound);
                }
            }
            WorkerPool::Thread(thread_workers) => {
                thread_workers.collect_rollout(bound);
            }
        }
    }

    pub fn single_step(&mut self) {
        self.collect(RolloutMode::StepBound { n_steps: 1 });
    }

    pub fn shutdown(&mut self) {
        match self {
            WorkerPool::Vec(_) => {
                // No need to explicitly shut down
            }
            WorkerPool::Thread(workers) => {
                workers.shutdown();
            }
        }
    }

    pub fn reset_all_envs(&mut self) {
        match self {
            WorkerPool::Vec(workers) => {
                for worker in workers {
                    let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
                    worker.reset(seed);
                }
            }
            WorkerPool::Thread(workers) => {
                workers.reset_all();
            }
        }
    }
}

impl<E: Env, B: ExpandableTrajectoryContainer<Tensor = <E as Env>::Tensor>> Drop
    for WorkerPool<E, B>
{
    fn drop(&mut self) {
        self.shutdown();
    }
}
