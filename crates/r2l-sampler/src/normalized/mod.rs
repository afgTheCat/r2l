// Fun programming. This is very reminacsent to the the VecNormalize in sb3
//
// Idea: implement a direct way of normalizing the environment. Workers not have access to the same
// buffer, instead they return the observation and the reward directly. Normalizaiton happens
// afterwards.

mod clipped_noramlizer;
mod worker;

use std::marker::PhantomData;

use bimodal_array::{
    ArrayHandle, ElementHandle, ElementWorker, ElementWorkerFactory, bimodal_array,
    bimodal_array_with_factory,
};
use crossbeam::channel::{Receiver, Sender};
use r2l_core::{
    buffers::buffer::{TrajectoryBuffer, TrajectoryView},
    buffers::{Memory, MultiMemory},
    env::{Env, EnvBuilder, EnvBuilderType, Snapshot},
    models::Actor,
    on_policy::algorithm::Sampler,
    rng::RNG,
    tensor::R2lTensor,
};
use rand::RngExt;

use crate::{
    SamplerExecutionMode,
    normalized::{
        clipped_noramlizer::ClippedNormalizer,
        worker::{WorkerCommand, WorkerResult},
    },
};

pub struct R2lNormalizedSampler<E: Env<Tensor: R2lTensor>> {
    pool: NewWorkerPool<E>,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    reward_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    last_states: ArrayHandle<E::Tensor>,
    // Here there is no need to have each thread own the buffer
    buffers: Vec<TrajectoryBuffer<E::Tensor>>,
    // TODO: we might want later on. Maybe other things?
    n_steps: usize,
}

struct NewWorker<T: R2lTensor, E: Env<Tensor = T>> {
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<T: R2lTensor, E: Env<Tensor = T>> NewWorker<T, E> {
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

struct NewVecWorker<T: R2lTensor, E: Env<Tensor = T>> {
    worker: NewWorker<T, E>,
    handle: bimodal_array::ElementHandle<T>,
}

impl<T: R2lTensor, E: Env<Tensor = T>> NewVecWorker<T, E> {
    fn step(&mut self) -> Memory<T> {
        self.worker.step(&mut self.handle)
    }

    fn set_policy(&mut self, policy: Box<dyn Actor<Tensor = T>>) {
        self.worker.actor = Some(policy);
    }
}

struct NewThreadWorker<T: R2lTensor, E: Env<Tensor = T>> {
    worker: NewWorker<T, E>,
    rx: Receiver<WorkerCommand<T>>,
    tx: Sender<WorkerResult<T>>,
}

impl<T: R2lTensor, E: Env<Tensor = T>> NewThreadWorker<T, E> {
    fn new(env: E, rx: Receiver<WorkerCommand<T>>, tx: Sender<WorkerResult<T>>) -> Self {
        Self {
            worker: NewWorker { actor: None, env },
            rx,
            tx,
        }
    }
}

impl<T: R2lTensor, E: Env<Tensor = T>> ElementWorker for NewThreadWorker<T, E> {
    type T = T;

    fn build(&mut self) -> Self::T {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        self.worker.env.reset(seed).unwrap()
    }

    fn work(&mut self, mut handle: ElementHandle<Self::T>) {
        while let Ok(command) = self.rx.recv() {
            match command {
                WorkerCommand::Step(_) => {
                    let memory = self.worker.step(&mut handle);
                    self.tx.send(WorkerResult::Stepped(memory)).unwrap();
                }
                WorkerCommand::SetPolicy(policy) => {
                    self.worker.actor = Some(policy);
                    self.tx.send(WorkerResult::PolicySet).unwrap();
                }
            }
        }
    }
}

struct NewThreadWorkerFactory<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> {
    rx: Receiver<WorkerCommand<T>>,
    tx: Sender<WorkerResult<T>>,
    env_builder: EB,
}

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> NewThreadWorkerFactory<T, EB> {
    fn new(rx: Receiver<WorkerCommand<T>>, tx: Sender<WorkerResult<T>>, env_builder: EB) -> Self {
        Self {
            rx,
            tx,
            env_builder,
        }
    }
}

impl<T: R2lTensor, EB: EnvBuilder<Env: Env<Tensor = T>>> ElementWorkerFactory
    for NewThreadWorkerFactory<T, EB>
{
    type Worker = NewThreadWorker<T, <EB as EnvBuilder>::Env>;

    fn build(self) -> Self::Worker {
        let env = self.env_builder.build_env().unwrap();
        NewThreadWorker::new(env, self.rx, self.tx)
    }
}

struct NewVecWorkers<T: R2lTensor, E: Env<Tensor = T>> {
    workers: Vec<NewVecWorker<T, E>>,
}

impl<T: R2lTensor, E: Env<Tensor = T>> NewVecWorkers<T, E> {
    fn step(&mut self) -> MultiMemory<T> {
        let mut multi_memory = MultiMemory::with_capacity(self.workers.len());
        for worker in &mut self.workers {
            multi_memory.push_memory(worker.step());
        }
        multi_memory
    }

    fn set_policy<A: Actor<Tensor = T> + Clone>(&mut self, policy: A) {
        for worker in &mut self.workers {
            worker.set_policy(Box::new(policy.clone()));
        }
    }
}

struct NewThreadWorkers<T: R2lTensor> {
    worker_count: usize,
    command_tx: Sender<WorkerCommand<T>>,
    result_rx: Receiver<WorkerResult<T>>,
}

impl<T: R2lTensor> NewThreadWorkers<T> {
    fn step(&self) -> MultiMemory<T> {
        for _ in 0..self.worker_count {
            self.command_tx
                .send(WorkerCommand::Step(PhantomData))
                .unwrap();
        }
        let mut multi_memory = MultiMemory::with_capacity(self.worker_count);
        for _ in 0..self.worker_count {
            let WorkerResult::Stepped(memory) = self.result_rx.recv().unwrap() else {
                unreachable!()
            };
            multi_memory.push_memory(memory);
        }
        multi_memory
    }

    fn set_policy<A: Actor<Tensor = T> + Clone>(&self, policy: A) {
        for _ in 0..self.worker_count {
            self.command_tx
                .send(WorkerCommand::SetPolicy(Box::new(policy.clone())))
                .unwrap();
        }
        for _ in 0..self.worker_count {
            let WorkerResult::PolicySet = self.result_rx.recv().unwrap() else {
                unreachable!()
            };
        }
    }
}

enum NewWorkerPool<E: Env<Tensor: R2lTensor>> {
    Vec(NewVecWorkers<E::Tensor, E>),
    Thread(NewThreadWorkers<E::Tensor>),
}

impl<E: Env<Tensor: R2lTensor>> NewWorkerPool<E> {
    fn step(&mut self) -> MultiMemory<E::Tensor> {
        match self {
            Self::Vec(workers) => workers.step(),
            Self::Thread(workers) => workers.step(),
        }
    }

    fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        match self {
            Self::Vec(workers) => workers.set_policy(policy),
            Self::Thread(workers) => workers.set_policy(policy),
        }
    }
}

impl<E: Env<Tensor: R2lTensor>> R2lNormalizedSampler<E> {
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        n_steps: usize,
        execution_mode: SamplerExecutionMode,
        _with_obs_normalizer: bool,
        _with_reward_normalizer: bool,
    ) -> Self {
        let num_envs = env_builder.num_envs();
        let buffers = vec![TrajectoryBuffer::default(); num_envs];
        let (last_states, pool) = match execution_mode {
            SamplerExecutionMode::Vec => Self::build_vec_workers(env_builder, num_envs),
            SamplerExecutionMode::Thread => Self::build_thread_workers(env_builder, num_envs),
        };
        Self {
            buffers,
            pool,
            last_states,
            obs_normalizer: None,
            reward_normalizer: None,
            n_steps,
        }
    }

    // TODO: ugly! will need to make this nicer
    fn build_vec_workers<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        num_envs: usize,
    ) -> (ArrayHandle<E::Tensor>, NewWorkerPool<E>) {
        let mut envs_and_states = Vec::with_capacity(num_envs);
        for env_idx in 0..num_envs {
            let mut env = env_builder.build_idx(env_idx).unwrap();
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            let state = env.reset(seed).unwrap();
            envs_and_states.push((env, state));
        }
        let initial_states = envs_and_states
            .iter()
            .map(|(_, state)| state.clone())
            .collect();
        let (last_states, last_state_handles) = bimodal_array(initial_states);
        let workers = envs_and_states
            .into_iter()
            .zip(last_state_handles)
            .map(|((env, _), handle)| NewVecWorker {
                worker: NewWorker { actor: None, env },
                handle,
            })
            .collect();
        (last_states, NewWorkerPool::Vec(NewVecWorkers { workers }))
    }

    fn build_thread_workers<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        num_envs: usize,
    ) -> (ArrayHandle<E::Tensor>, NewWorkerPool<E>) {
        let (command_tx, command_rx) = crossbeam::channel::unbounded();
        let (result_tx, result_rx) = crossbeam::channel::unbounded();
        let factories = (0..num_envs)
            .map(|idx| {
                let env_builder = env_builder.clone();
                let env_builder = move || env_builder.build_idx(idx);
                NewThreadWorkerFactory::new(
                    command_rx.clone(),
                    result_tx.clone(),
                    env_builder.clone(),
                )
            })
            .collect();
        let last_states = bimodal_array_with_factory(factories);
        let workers = NewThreadWorkers {
            worker_count: num_envs,
            command_tx,
            result_rx,
        };
        (last_states, NewWorkerPool::Thread(workers))
    }

    fn step(&mut self) {
        let mut multi_memory = self.pool.step();
        multi_memory.next_states = if let Some(obs_normalizer) = self.obs_normalizer.as_mut() {
            let next_states = std::mem::take(&mut multi_memory.next_states);
            let normalized_next_states = obs_normalizer.update_and_normalize(&next_states);
            for (last_state, normalized_next_state) in self
                .last_states
                .lock()
                .unwrap()
                .iter_mut()
                .zip(normalized_next_states.iter())
            {
                *last_state = normalized_next_state.clone();
            }
            normalized_next_states
        } else {
            std::mem::take(&mut multi_memory.next_states)
        };

        // TODO: add this once the normalizer is working as intended
        // multi_memory.rewards = if let Some(rew_normalizer) = self.reward_normalizer.as_mut() {
        //     rew_normalizer.normalize(std::mem::take(&mut multi_memory.rewards))
        // } else {
        //     std::mem::take(&mut multi_memory.rewards)
        // };
        let memories = multi_memory.into_memories();
        for (idx, memory) in memories.into_iter().enumerate() {
            self.buffers[idx].push(memory);
        }
    }
}

impl<E: Env<Tensor: R2lTensor>> Sampler for R2lNormalizedSampler<E> {
    type Tensor = E::Tensor;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        self.buffers.iter_mut().for_each(|b| b.clear());
        self.pool.set_policy(actor.clone());
        let mut steps = 0;
        while steps < self.n_steps {
            self.step();
            steps += 1;
        }
    }

    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]> {
        self.buffers
            .iter()
            .map(|b| b.to_trajectory_view())
            .collect::<Vec<_>>()
    }

    fn shutdown(&mut self) {}
}
