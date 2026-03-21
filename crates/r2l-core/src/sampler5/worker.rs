use std::collections::HashMap;

use crate::{
    distributions::Policy,
    env::{Env, EnvironmentDescription, SnapShot},
    rng::RNG,
    sampler5::{
        RolloutMode,
        buffer::{ExpandableTrajectoryContainer, Memory},
    },
    tensor::R2lTensor,
};
use bimodal_array::ElementHandle;
use crossbeam::channel::{Receiver, Sender};
use rand::Rng;

pub struct Worker<E: Env, D: ExpandableTrajectoryContainer<Tensor = E::Tensor>> {
    pub env: E,
    pub buffer: ElementHandle<D>,
    // TODO: this is a bit archaic, we might want somethings else
    pub policy: Option<Box<dyn Policy<Tensor = E::Tensor>>>,
    pub last_state: Option<E::Tensor>,
}

fn step_env<T: R2lTensor, E: Env<Tensor = T>>(
    env: &mut E,
    distr: &mut Box<dyn Policy<Tensor = T>>,
    last_state: Option<T>,
) -> Memory<T> {
    let state = if let Some(state) = last_state {
        state
    } else {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        env.reset(seed).unwrap()
    };

    let action = distr.get_action(state.clone()).unwrap();
    let SnapShot {
        state: mut next_state,
        reward,
        terminated,
        trancuated,
    } = env.step(action.clone()).unwrap();
    let done = terminated || trancuated;
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
        trancuated,
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
        let Some(distr) = &mut self.policy else {
            todo!()
        };
        let mut buffer = self.buffer.lock().unwrap();
        match bound {
            RolloutMode::StepBound { n_steps: steps } => {
                for _ in 0..steps {
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, distr, last_state);
                    self.last_state = Some(memory.next_state.clone());
                    buffer.push(memory);
                }
            }
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = 0;
                loop {
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, distr, last_state);
                    let terminates = memory.terminates();
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
}

pub enum WorkerCommand<T: R2lTensor> {
    SetPolicy(Box<dyn Policy<Tensor = T>>),
    Collect(RolloutMode),
}

pub enum WorkerResult {
    PolicySet,
    Collected,
}

pub struct ThreadWorker<E: Env, D: ExpandableTrajectoryContainer<Tensor = E::Tensor>> {
    worker: Worker<E, D>,
    rx: Receiver<WorkerCommand<E::Tensor>>,
    tx: Sender<WorkerResult>,
}

impl<E: Env, D: ExpandableTrajectoryContainer<Tensor = E::Tensor>> ThreadWorker<E, D> {
    pub fn new(
        worker: Worker<E, D>,
        rx: Receiver<WorkerCommand<E::Tensor>>,
        tx: Sender<WorkerResult>,
    ) -> Self {
        Self { worker, rx, tx }
    }

    pub fn work(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                WorkerCommand::SetPolicy(distr) => {
                    self.worker.policy = Some(distr);
                    self.tx.send(WorkerResult::PolicySet).unwrap();
                }
                WorkerCommand::Collect(bound) => {
                    self.worker.collect_rollout(bound);
                }
            }
        }
    }
}

pub struct ThreadWorkers<T: R2lTensor>(
    pub HashMap<usize, (Sender<WorkerCommand<T>>, Receiver<WorkerResult>)>,
);

impl<T: R2lTensor> ThreadWorkers<T> {
    pub fn set_policy<P: Policy<Tensor = T> + Clone>(&self, policy: P) {
        let channels = &self.0;
        let num_envs = channels.len();
        for idx in 0..num_envs {
            let tx = &channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::SetPolicy(Box::new(policy.clone())))
                .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    pub fn collect_rollout(&self, bound: RolloutMode) {
        let channels = &self.0;
        let num_envs = channels.len();
        for idx in 0..num_envs {
            let tx = &channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::Collect(bound)).unwrap();
        }
        for idx in 0..num_envs {
            let rx = &channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }
}

pub enum WorkerPool<E: Env, B: ExpandableTrajectoryContainer<Tensor = <E as Env>::Tensor>> {
    Vec(Vec<Worker<E, B>>),
    Thread(ThreadWorkers<E::Tensor>),
}

impl<E: Env, B: ExpandableTrajectoryContainer<Tensor = <E as Env>::Tensor>> WorkerPool<E, B> {
    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        match self {
            Self::Vec(workers) => workers[0].env.env_description(),
            _ => todo!(),
        }
    }

    pub fn set_policy<P: Policy<Tensor = E::Tensor> + Clone>(&mut self, policy: P) {
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
}
