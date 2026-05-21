use std::ops::Bound;

use bimodal_array::{ArrayHandle, ElementHandle};
use r2l_core::{
    buffers::buffer::NewBuffer,
    env::{Env, EnvDescription},
    models::Actor,
    on_policy::algorithm::Sampler,
    rng::RNG,
    tensor::R2lTensor,
};
use rand::RngExt;

use crate::{
    RolloutBound, RolloutMode,
    worker::{CommandReceiver, ResultSender, ThreadHandle, WorkerCommand, WorkerResult, step_env},
};

// The new worker thingy
pub struct Worker2<E: Env> {
    pub env: E,
    pub buffer: ElementHandle<NewBuffer<E::Tensor>>,
    pub policy: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    pub last_state: Option<E::Tensor>,
}

impl<E: Env> Worker2<E> {
    pub fn new(env: E, buffer: ElementHandle<NewBuffer<E::Tensor>>) -> Self {
        Self {
            env,
            buffer,
            policy: None,
            last_state: None,
        }
    }

    pub fn collect(&mut self, bound: RolloutMode) {
        let Some(policy) = &mut self.policy else {
            todo!()
        };
        let mut buffer = self.buffer.lock().unwrap();
        match bound {
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = 0;
                loop {
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
            RolloutMode::StepBound { n_steps } => {
                for _ in 0..n_steps {
                    let last_state = self.last_state.take();
                    let memory = step_env(&mut self.env, policy, last_state);
                    self.last_state = Some(memory.next_state.clone());
                    buffer.push(memory);
                }
            }
        }
    }

    pub fn reset(&mut self, seed: u64) {
        let state = self.env.reset(seed).unwrap();
        self.last_state = Some(state);
        // TODO: do we need to force reset the buffer?
    }
}

pub struct ThreadWorker2<E: Env> {
    worker: Worker2<E>,
    rx: CommandReceiver<E::Tensor>,
    tx: ResultSender<E::Tensor>,
}

impl<E: Env> ThreadWorker2<E> {
    pub fn new(
        worker: Worker2<E>,
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
            }
        }
    }
}

pub struct ThreadWorkers2<T: R2lTensor> {
    worker_handles: Vec<ThreadHandle<T>>,
}

impl<T: R2lTensor> ThreadWorkers2<T> {
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

pub enum WorkerPool2<E: Env> {
    Vec(Vec<Worker2<E>>),
    Thread(ThreadWorkers2<E::Tensor>),
}

impl<E: Env> WorkerPool2<E> {
    pub fn env_description(&self) -> EnvDescription<E::Tensor> {
        match self {
            Self::Vec(workers) => workers[0].env.env_description(),
            Self::Thread(tw) => tw.env_description(),
        }
    }

    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, policy: A) {
        match self {
            Self::Vec(workers) => {
                for worker in workers.iter_mut() {
                    worker.policy = Some(Box::new(policy.clone()))
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
                    let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
                    worker.reset(seed);
                }
            }
            Self::Thread(workers) => {
                workers.reset_all();
            }
        }
    }
}

impl<E: Env> Drop for WorkerPool2<E> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

enum SamplerHookResult {
    Stop,
    Bound(RolloutMode),
}

trait SamplerHook {
    type E: Env;

    fn hook(
        &mut self,
        buffer: &mut ArrayHandle<NewBuffer<<Self::E as Env>::Tensor>>,
    ) -> SamplerHookResult;
}

pub struct R2lSampler2<E: Env, H: SamplerHook<E = E>> {
    buffer: ArrayHandle<NewBuffer<E::Tensor>>,
    worker_pool: WorkerPool2<E>,
    hook: H,
}

impl<E: Env, H: SamplerHook<E = E>> R2lSampler2<E, H> {
    fn collect_rollouts2<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, actor: A) {
        loop {
            let result = self.hook.hook(&mut self.buffer);
            match result {
                SamplerHookResult::Bound(bound) => self.worker_pool.collect(bound),
                SamplerHookResult::Stop => break,
            }
        }
    }
}
