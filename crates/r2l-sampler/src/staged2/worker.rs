use bimodal_array::{
    ArrayGuard, ArrayHandle, ElementHandle, ElementWorker, ElementWorkerFactory, bimodal_array,
    bimodal_array_with_factory,
};
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::Memory,
    env::{Env, EnvBuilder, EnvBuilderType, Snapshot},
    models::Actor,
    rng::{sample_u64, set_seed},
};

use crate::SamplerExecutionMode;

enum WorkerCommand<T> {
    Step,
    SetPolicy(Box<dyn Actor<Tensor = T>>),
    ResetEnv(u64),
    Stop,
}

enum WorkerResult<T> {
    Stepped(Memory<T>),
    PolicySet,
    EnvReset,
    Stopped,
}

struct Worker<E: Env> {
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<E: Env> Worker<E> {
    fn new(env: E) -> Self {
        Self { actor: None, env }
    }

    fn step(&mut self, state_handle: &mut ElementHandle<E::Tensor>) -> Memory<E::Tensor> {
        let actor = self.actor.as_mut().expect("worker policy is not set");
        let state = state_handle.lock().unwrap().clone();
        let action = actor.action(state.clone()).unwrap();
        let Snapshot {
            state: next_state,
            reward,
            terminated,
            truncated,
        } = self.env.step(action.clone()).unwrap();
        *state_handle.lock().unwrap() = next_state.clone();
        Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        }
    }

    fn set_policy(&mut self, policy: Box<dyn Actor<Tensor = E::Tensor>>) {
        self.actor = Some(policy);
    }

    fn reset(&mut self, state_handle: &mut ElementHandle<E::Tensor>, seed: u64) {
        *state_handle.lock().unwrap() = self.env.reset(seed).unwrap();
    }
}

struct VecWorker<E: Env> {
    worker: Worker<E>,
    state_handle: ElementHandle<E::Tensor>,
}

impl<E: Env> VecWorker<E> {
    fn new(env: E, state_handle: ElementHandle<E::Tensor>) -> Self {
        Self {
            worker: Worker::new(env),
            state_handle,
        }
    }

    fn step(&mut self) -> Memory<E::Tensor> {
        self.worker.step(&mut self.state_handle)
    }

    fn reset(&mut self, seed: u64) {
        self.worker.reset(&mut self.state_handle, seed);
    }
}

struct VecWorkers<E: Env> {
    workers: Vec<VecWorker<E>>,
}

impl<E: Env> VecWorkers<E> {
    fn new(workers: Vec<(E, ElementHandle<E::Tensor>)>) -> Self {
        Self {
            workers: workers
                .into_iter()
                .map(|(env, state_handle)| VecWorker::new(env, state_handle))
                .collect(),
        }
    }

    fn step(&mut self) -> Vec<Memory<E::Tensor>> {
        self.workers.iter_mut().map(VecWorker::step).collect()
    }

    fn step_indexed(&mut self, indices: &[usize]) -> Vec<Memory<E::Tensor>> {
        indices
            .iter()
            .map(|idx| self.workers[*idx].step())
            .collect()
    }

    fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        self.workers
            .iter_mut()
            .for_each(|worker| worker.worker.set_policy(Box::new(policy.clone())));
    }

    fn reset_indexed(&mut self, indices: &[usize]) {
        indices
            .iter()
            .for_each(|idx| self.workers[*idx].reset(sample_u64()));
    }
}

struct ThreadWorker<E: Env> {
    worker: Worker<E>,
    command_rx: Receiver<WorkerCommand<E::Tensor>>,
    result_tx: Sender<WorkerResult<E::Tensor>>,
}

impl<E: Env> ThreadWorker<E> {
    fn new(
        env: E,
        command_rx: Receiver<WorkerCommand<E::Tensor>>,
        result_tx: Sender<WorkerResult<E::Tensor>>,
    ) -> Self {
        Self {
            worker: Worker::new(env),
            command_rx,
            result_tx,
        }
    }
}

impl<E: Env> ElementWorker for ThreadWorker<E> {
    type T = E::Tensor;

    fn build(&mut self) -> Self::T {
        self.worker.env.reset(sample_u64()).unwrap()
    }

    fn work(&mut self, mut state_handle: ElementHandle<Self::T>) {
        while let Ok(command) = self.command_rx.recv() {
            match command {
                WorkerCommand::Step => {
                    let memory = self.worker.step(&mut state_handle);
                    self.result_tx.send(WorkerResult::Stepped(memory)).unwrap();
                }
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.set_policy(policy);
                    self.result_tx.send(WorkerResult::PolicySet).unwrap();
                }
                WorkerCommand::ResetEnv(seed) => {
                    self.worker.reset(&mut state_handle, seed);
                    self.result_tx.send(WorkerResult::EnvReset).unwrap();
                }
                WorkerCommand::Stop => {
                    self.result_tx.send(WorkerResult::Stopped).unwrap();
                    break;
                }
            }
        }
    }
}

struct ThreadWorkerFactory<EB: EnvBuilder> {
    command_rx: Receiver<WorkerCommand<<EB::Env as Env>::Tensor>>,
    result_tx: Sender<WorkerResult<<EB::Env as Env>::Tensor>>,
    env_builder: EB,
    worker_seed: u64,
}

impl<EB: EnvBuilder> ThreadWorkerFactory<EB> {
    fn new(
        command_rx: Receiver<WorkerCommand<<EB::Env as Env>::Tensor>>,
        result_tx: Sender<WorkerResult<<EB::Env as Env>::Tensor>>,
        env_builder: EB,
        worker_seed: u64,
    ) -> Self {
        Self {
            command_rx,
            result_tx,
            env_builder,
            worker_seed,
        }
    }
}

impl<EB: EnvBuilder> ElementWorkerFactory for ThreadWorkerFactory<EB> {
    type Worker = ThreadWorker<EB::Env>;

    fn build(self) -> Self::Worker {
        set_seed(self.worker_seed);
        ThreadWorker::new(
            self.env_builder.build_env().unwrap(),
            self.command_rx,
            self.result_tx,
        )
    }
}

struct ThreadHandle<T> {
    command_tx: Sender<WorkerCommand<T>>,
    result_rx: Receiver<WorkerResult<T>>,
}

impl<T> ThreadHandle<T> {
    fn new(command_tx: Sender<WorkerCommand<T>>, result_rx: Receiver<WorkerResult<T>>) -> Self {
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

struct ThreadWorkers<T> {
    worker_handles: Vec<ThreadHandle<T>>,
}

impl<T> ThreadWorkers<T> {
    fn new(worker_handles: Vec<ThreadHandle<T>>) -> Self {
        Self { worker_handles }
    }

    fn step(&self) -> Vec<Memory<T>> {
        self.step_indexed(&(0..self.worker_handles.len()).collect::<Vec<_>>())
    }

    fn step_indexed(&self, indices: &[usize]) -> Vec<Memory<T>> {
        indices
            .iter()
            .for_each(|idx| self.worker_handles[*idx].send(WorkerCommand::Step));
        indices
            .iter()
            .map(|idx| {
                let WorkerResult::Stepped(memory) = self.worker_handles[*idx].recv() else {
                    unreachable!()
                };
                memory
            })
            .collect()
    }

    fn set_policy<A: Actor<Tensor = T> + Clone>(&self, policy: A) {
        self.worker_handles.iter().for_each(|worker| {
            worker.send(WorkerCommand::SetPolicy(Box::new(policy.clone())));
        });
        self.worker_handles.iter().for_each(|worker| {
            let WorkerResult::PolicySet = worker.recv() else {
                unreachable!()
            };
        });
    }

    fn reset_indexed(&self, indices: &[usize]) {
        indices.iter().for_each(|idx| {
            self.worker_handles[*idx].send(WorkerCommand::ResetEnv(sample_u64()));
        });
        indices.iter().for_each(|idx| {
            let WorkerResult::EnvReset = self.worker_handles[*idx].recv() else {
                unreachable!()
            };
        });
    }

    fn shutdown(&self) {
        self.worker_handles
            .iter()
            .for_each(|worker| worker.send(WorkerCommand::Stop));
        self.worker_handles.iter().for_each(|worker| {
            let WorkerResult::Stopped = worker.recv() else {
                unreachable!()
            };
        });
    }
}

enum WorkerExecution2<E: Env> {
    Vec(VecWorkers<E>),
    Thread(ThreadWorkers<E::Tensor>),
}

/// Environment workers with owned access to their shared current observations.
pub struct WorkerPool2<E: Env> {
    workers: WorkerExecution2<E>,
    pub last_states: ArrayHandle<E::Tensor>,
    num_envs: usize,
}

impl<E: Env> WorkerPool2<E> {
    /// Builds a raw-observation worker pool.
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        match execution_mode {
            SamplerExecutionMode::SingleThreaded => Self::build_vec(env_builder),
            SamplerExecutionMode::MultiThreaded => Self::build_threaded(env_builder),
        }
    }

    fn build_vec<EB: EnvBuilder<Env = E>>(env_builder: EnvBuilderType<EB>) -> Self {
        let num_envs = env_builder.num_envs();
        let mut envs = Vec::with_capacity(num_envs);
        let mut initial_states = Vec::with_capacity(num_envs);
        for idx in 0..num_envs {
            let mut env = env_builder.build_idx(idx).unwrap();
            initial_states.push(env.reset(sample_u64()).unwrap());
            envs.push(env);
        }
        let (last_states, state_handles) = bimodal_array(initial_states);
        let workers = VecWorkers::new(envs.into_iter().zip(state_handles).collect());
        Self {
            workers: WorkerExecution2::Vec(workers),
            last_states,
            num_envs,
        }
    }

    fn build_threaded<EB: EnvBuilder<Env = E>>(env_builder: EnvBuilderType<EB>) -> Self {
        let num_envs = env_builder.num_envs();
        let mut worker_handles = Vec::with_capacity(num_envs);
        let factories = (0..num_envs)
            .map(|idx| {
                let (command_tx, command_rx) = crossbeam::channel::unbounded();
                let (result_tx, result_rx) = crossbeam::channel::unbounded();
                worker_handles.push(ThreadHandle::new(command_tx, result_rx));
                let env_builder = env_builder.clone();
                let builder = move || env_builder.build_idx(idx);
                ThreadWorkerFactory::new(command_rx, result_tx, builder, sample_u64())
            })
            .collect();
        Self {
            workers: WorkerExecution2::Thread(ThreadWorkers::new(worker_handles)),
            last_states: bimodal_array_with_factory(factories),
            num_envs,
        }
    }

    /// Steps every worker once and returns complete raw-observation transitions.
    pub fn step(&mut self) -> Vec<Memory<E::Tensor>> {
        match &mut self.workers {
            WorkerExecution2::Vec(workers) => workers.step(),
            WorkerExecution2::Thread(workers) => workers.step(),
        }
    }

    /// Steps selected workers and returns transitions in `indices` order.
    pub fn step_indexed(&mut self, indices: &[usize]) -> Vec<Memory<E::Tensor>> {
        match &mut self.workers {
            WorkerExecution2::Vec(workers) => workers.step_indexed(indices),
            WorkerExecution2::Thread(workers) => workers.step_indexed(indices),
        }
    }

    /// Installs a clone of `policy` on every worker.
    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        match &mut self.workers {
            WorkerExecution2::Vec(workers) => workers.set_policy(policy),
            WorkerExecution2::Thread(workers) => workers.set_policy(policy),
        }
    }

    /// Resets every environment.
    pub fn reset_all(&mut self) {
        let indices = (0..self.len()).collect::<Vec<_>>();
        self.reset_indexed(&indices);
    }

    /// Resets selected environments.
    pub fn reset_indexed(&mut self, indices: &[usize]) {
        match &mut self.workers {
            WorkerExecution2::Vec(workers) => workers.reset_indexed(indices),
            WorkerExecution2::Thread(workers) => workers.reset_indexed(indices),
        }
    }

    /// Returns the number of environments.
    pub fn len(&self) -> usize {
        self.num_envs
    }

    /// Returns whether the pool contains no environments.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clones current observations in worker order.
    pub fn current_states(&mut self) -> Vec<E::Tensor> {
        self.last_states.lock().unwrap().to_vec()
    }

    /// Stops threaded workers.
    pub fn shutdown(&mut self) {
        if let WorkerExecution2::Thread(workers) = &self.workers {
            workers.shutdown();
        }
    }

    pub fn last_state_mut<'a>(&'a mut self) -> ArrayGuard<'a, E::Tensor> {
        self.last_states.lock().unwrap()
    }
}
