use std::thread::JoinHandle;

use bimodal_array::ElementHandle;
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::{Memory, buffer::TrajectoryBuffer},
    env::{Env, EnvDescription, Snapshot},
    models::Actor,
    rng::sample_u64,
    tensor::R2lTensor,
};

use crate::direct::RolloutMode;

pub(crate) type CommandSender<T, S> = Sender<WorkerCommand<T, S>>;
pub(crate) type CommandReceiver<T, S> = Receiver<WorkerCommand<T, S>>;

pub(crate) type ResultSender<T> = Sender<WorkerResult<T>>;
pub(crate) type ResultReceiver<T> = Receiver<WorkerResult<T>>;

pub(crate) fn step_env<T, E, S>(
    env: &mut E,
    actor: &dyn Actor<Tensor = T, State = S>,
    actor_state: Option<S>,
    last_observation: Option<T>,
) -> (Memory<T, S>, Option<S>)
where
    T: R2lTensor,
    E: Env<Tensor = T>,
    S: Clone + Send + Sync + 'static,
{
    let state = if let Some(observation) = last_observation {
        observation
    } else {
        env.reset(sample_u64()).unwrap()
    };
    let memory_actor_state = actor_state.clone();
    let (action, next_actor_state) = actor.action(state.clone(), actor_state).unwrap();
    let Snapshot {
        state: mut next_state,
        reward,
        terminated,
        truncated,
    } = env.step(action.clone()).unwrap();
    let done = terminated || truncated;
    if done {
        next_state = env.reset(sample_u64()).unwrap();
    }
    (
        Memory {
            state,
            next_state,
            action,
            actor_state: memory_actor_state,
            reward,
            terminated,
            truncated,
        },
        (!done).then_some(next_actor_state),
    )
}

pub(crate) enum WorkerCommand<T: R2lTensor, S: Clone + Send + Sync + 'static> {
    SetPolicy(Box<dyn Actor<Tensor = T, State = S>>),
    Collect(RolloutMode),
    ResetEnv(u64),
    ClearBuffer,
    GetEnvDescription,
    Shutdown,
    GetLastObservation,
    SetLastObservation(T),
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
    LastObservation(Option<T>),
    LastObservationSet,
    ResetEnvUninsertedResult(T),
    LastNextStateReplaced,
}

pub struct ThreadHandle<T: R2lTensor, S: Clone + Send + Sync + 'static> {
    handle: JoinHandle<()>,
    command_tx: CommandSender<T, S>,
    worker_rx: ResultReceiver<T>,
}

impl<T: R2lTensor, S: Clone + Send + Sync + 'static> ThreadHandle<T, S> {
    pub(crate) fn new(
        handle: JoinHandle<()>,
        command_tx: CommandSender<T, S>,
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

    pub(crate) fn send(&self, command: WorkerCommand<T, S>) {
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

pub struct Worker<E: Env, S: Clone + Send + Sync + 'static = ()> {
    pub env: E,
    pub buffer: ElementHandle<TrajectoryBuffer<E::Tensor, S>>,
    actor: Option<Box<dyn Actor<Tensor = E::Tensor, State = S>>>,
    actor_state: Option<S>,
    pub last_observation: Option<E::Tensor>,
}

impl<E: Env, S: Clone + Send + Sync + 'static> Worker<E, S> {
    pub fn new(env: E, buffer: ElementHandle<TrajectoryBuffer<E::Tensor, S>>) -> Self {
        Self {
            env,
            buffer,
            actor: None,
            actor_state: None,
            last_observation: None,
        }
    }

    fn set_actor(&mut self, actor: Box<dyn Actor<Tensor = E::Tensor, State = S>>) {
        self.actor = Some(actor);
    }

    pub fn set_last_observation(&mut self, observation: E::Tensor) {
        self.last_observation = Some(observation);
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
        let Some(actor) = &self.actor else { todo!() };
        let mut buffer = self.buffer.lock().unwrap();
        match bound {
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = 0;
                loop {
                    let actor_state = self.actor_state.take();
                    let last_observation = self.last_observation.take();
                    let (memory, actor_state) =
                        step_env(&mut self.env, actor.as_ref(), actor_state, last_observation);
                    let terminates = memory.is_done();
                    self.actor_state = actor_state;
                    self.last_observation = Some(memory.next_state.clone());
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
                    let actor_state = self.actor_state.take();
                    let last_observation = self.last_observation.take();
                    let (memory, actor_state) =
                        step_env(&mut self.env, actor.as_ref(), actor_state, last_observation);
                    self.actor_state = actor_state;
                    self.last_observation = Some(memory.next_state.clone());
                    buffer.push(memory);
                }
            }
        }
    }

    // resets the initial state and clears the buffer. Used by the Evaluator hook
    pub fn reset(&mut self, seed: u64) {
        let state = self.env.reset(seed).unwrap();
        self.actor_state = None;
        self.last_observation = Some(state);
        self.buffer.lock().unwrap().clear();
    }

    pub fn reset_env_uninserted(&mut self, seed: u64) -> E::Tensor {
        self.env.reset(seed).unwrap()
    }
}

pub struct ThreadWorker<E: Env, S: Clone + Send + Sync + 'static = ()> {
    worker: Worker<E, S>,
    rx: CommandReceiver<E::Tensor, S>,
    tx: ResultSender<E::Tensor>,
}

impl<E: Env, S: Clone + Send + Sync + 'static> ThreadWorker<E, S> {
    pub fn new(
        worker: Worker<E, S>,
        rx: CommandReceiver<E::Tensor, S>,
        tx: ResultSender<E::Tensor>,
    ) -> Self {
        Self { worker, rx, tx }
    }

    pub fn work(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.set_actor(policy);
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
                WorkerCommand::GetLastObservation => {
                    let observation = self.worker.last_observation.clone();
                    self.tx
                        .send(WorkerResult::LastObservation(observation))
                        .unwrap();
                }
                WorkerCommand::SetLastObservation(observation) => {
                    self.worker.set_last_observation(observation);
                    self.tx.send(WorkerResult::LastObservationSet).unwrap();
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

pub struct ThreadWorkers<T: R2lTensor, S: Clone + Send + Sync + 'static = ()> {
    worker_handles: Vec<ThreadHandle<T, S>>,
}

impl<T: R2lTensor, S: Clone + Send + Sync + 'static> ThreadWorkers<T, S> {
    pub fn new(worker_handles: Vec<ThreadHandle<T, S>>) -> Self {
        Self { worker_handles }
    }

    // TODO: this can fail. We need to mark this as failible once we figured the right Error types out
    pub fn env_description(&self) -> EnvDescription<T> {
        self.worker_handles[0].env_description()
    }

    pub fn set_policy<A: Actor<Tensor = T, State = S> + Clone>(&self, policy: A) {
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
            worker_handle.send(WorkerCommand::ResetEnv(sample_u64()));
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }

    pub fn get_last_observations(&self) -> Option<Vec<T>> {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::GetLastObservation);
        }
        self.worker_handles
            .iter()
            .map(|h| {
                let WorkerResult::LastObservation(observation) = h.recv() else {
                    unreachable!()
                };
                observation
            })
            .collect()
    }

    pub fn set_last_observations(&self, observations: Vec<T>) {
        for (worker_handle, observation) in self.worker_handles.iter().zip(observations) {
            worker_handle.send(WorkerCommand::SetLastObservation(observation));
        }
        for worker_handle in self.worker_handles.iter() {
            worker_handle.recv();
        }
    }

    pub fn reset_envs_uninserted(&self) -> Vec<T> {
        for worker_handle in self.worker_handles.iter() {
            worker_handle.send(WorkerCommand::ResetEnvUninserted(sample_u64()));
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

pub enum WorkerPool<E: Env, S: Clone + Send + Sync + 'static = ()> {
    Vec(Vec<Worker<E, S>>),
    Thread(ThreadWorkers<E::Tensor, S>),
}

impl<E: Env, S: Clone + Send + Sync + 'static> WorkerPool<E, S> {
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

    pub fn set_actor<A: Actor<Tensor = E::Tensor, State = S> + Clone>(&mut self, policy: A) {
        match self {
            Self::Vec(workers) => {
                for worker in workers.iter_mut() {
                    worker.set_actor(Box::new(policy.clone()))
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
                    worker.reset(sample_u64());
                }
            }
            Self::Thread(workers) => {
                workers.reset_all();
            }
        }
    }

    pub fn get_last_observations(&mut self) -> Option<Vec<E::Tensor>> {
        match self {
            Self::Vec(workers) => {
                // in the order of the workers
                workers.iter().map(|w| w.last_observation.clone()).collect()
            }
            Self::Thread(workers) => {
                // worker pools ensures the order
                workers.get_last_observations()
            }
        }
    }

    pub fn set_last_observations(&mut self, observations: Vec<E::Tensor>) {
        match self {
            Self::Vec(workers) => {
                for (worker, observation) in workers.iter_mut().zip(observations) {
                    worker.set_last_observation(observation)
                }
            }
            Self::Thread(workers) => {
                workers.set_last_observations(observations);
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
                    .map(|w| w.reset_env_uninserted(sample_u64()))
                    .collect()
            }
            Self::Thread(workers) => workers.reset_envs_uninserted(),
        }
    }
}

impl<E: Env, S: Clone + Send + Sync + 'static> Drop for WorkerPool<E, S> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use r2l_core::{
        env::{Env, EnvDescription, Snapshot, Space},
        models::Actor,
        tensor::TensorData,
    };

    use super::step_env;

    struct CountingActor;

    impl Actor for CountingActor {
        type Tensor = TensorData;
        type State = usize;

        fn action(
            &self,
            observation: Self::Tensor,
            state: Option<Self::State>,
        ) -> Result<(Self::Tensor, Self::State)> {
            Ok((observation, state.unwrap_or_default() + 1))
        }
    }

    struct TestEnv;

    impl Env for TestEnv {
        type Tensor = TensorData;

        fn reset(&mut self, _seed: u64) -> Result<Self::Tensor> {
            Ok(TensorData::from_vec(vec![0.0]))
        }

        fn step(&mut self, action: Self::Tensor) -> Result<Snapshot<Self::Tensor>> {
            Ok(Snapshot::new(action, 0.0, false, false))
        }

        fn env_description(&self) -> EnvDescription<Self::Tensor> {
            EnvDescription::new(
                Space::Box {
                    min: None,
                    max: None,
                    shape: vec![1],
                },
                Space::Box {
                    min: None,
                    max: None,
                    shape: vec![1],
                },
            )
        }
    }

    #[test]
    fn step_env_carries_typed_actor_state() {
        let mut env = TestEnv;
        let actor = CountingActor;

        let (memory, state) = step_env(&mut env, &actor, None, None);
        assert_eq!(memory.actor_state, None);
        assert_eq!(state, Some(1));

        let (memory, state) = step_env(&mut env, &actor, state, Some(memory.next_state));
        assert_eq!(memory.actor_state, Some(1));
        assert_eq!(state, Some(2));
    }
}
