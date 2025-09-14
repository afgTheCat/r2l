pub mod builder;

use crate::{
    distributions::Policy,
    env::Env,
    sampler2::{Buffer, CollectionBound, EnvPool},
};
use crossbeam::channel::{Receiver, Sender};
use std::collections::HashMap;

pub enum WorkerCommand<E: Env> {
    SetPolicy(Box<dyn Policy<Tensor = E::Tensor>>),
    Collect(CollectionBound),
}

// what we can do, is have a buffer be on heap that both the worker and the main thread can reach
pub enum WorkerResult<B: Buffer> {
    PolicySet,
    Collected(B),
}

// TODO: better name here
pub struct ThreadEnvWorker<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Send + Clone> {
    tx: Sender<WorkerResult<B>>,
    rx: Receiver<WorkerCommand<E>>,
    buffer: B,
    env: E,
    policy: Option<Box<dyn Policy<Tensor = E::Tensor>>>,
    last_state: Option<E::Tensor>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Send + Clone> ThreadEnvWorker<E, B> {
    pub fn new(
        tx: Sender<WorkerResult<B>>,
        rx: Receiver<WorkerCommand<E>>,
        buffer: B,
        env: E,
    ) -> Self {
        Self {
            tx,
            rx,
            buffer,
            env,
            policy: None,
            last_state: None,
        }
    }

    pub fn step(&mut self) {
        let Some(distr) = &mut self.policy else {
            todo!()
        };
        let last_state = self.last_state.take();
        self.buffer.step(&mut self.env, distr, last_state);
    }

    pub fn collect(&mut self, bound: CollectionBound) {
        match bound {
            CollectionBound::StepBound { steps } => {
                for _ in 0..steps {
                    self.step();
                }
            }
            CollectionBound::EpisodeBound { steps } => {
                let mut steps_taken = 0;
                loop {
                    self.step();
                    steps_taken += 1;
                    if steps_taken >= steps && self.buffer.last_state_terminates() {
                        break;
                    }
                }
            }
        }
    }

    pub fn work(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                WorkerCommand::SetPolicy(distr) => {
                    self.policy = Some(distr);
                    self.tx.send(WorkerResult::PolicySet).unwrap();
                }
                WorkerCommand::Collect(bound) => {
                    self.collect(bound);
                    self.tx
                        .send(WorkerResult::Collected(self.buffer.clone()))
                        .unwrap();
                }
            }
        }
    }
}

pub type ChannelHashMap<E, B> =
    HashMap<usize, (Sender<WorkerCommand<E>>, Receiver<WorkerResult<B>>)>;

pub struct ThreadEnvPool<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> {
    pub collection_bound: CollectionBound,
    pub channels: ChannelHashMap<E, B>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> ThreadEnvPool<E, B> {
    pub fn num_envs(&self) -> usize {
        self.channels.len()
    }
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Send + Clone> EnvPool
    for ThreadEnvPool<E, B>
{
    type E = E;
    type B = B;

    fn collection_bound(&self) -> CollectionBound {
        self.collection_bound.clone()
    }

    fn set_policy<P: Policy<Tensor = <Self::E as Env>::Tensor> + Clone>(&mut self, policy: P) {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::SetPolicy(Box::new(policy.clone())))
                .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn get_buffers(&self) -> Vec<Self::B> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::Collect(CollectionBound::StepBound {
                steps: 0,
            }))
            .unwrap();
        }
        let mut buffers = vec![];
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let WorkerResult::Collected(buffer) = rx.recv().unwrap() else {
                panic!()
            };
            buffers.push(buffer);
        }
        buffers
    }

    fn single_step(&mut self) {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::Collect(CollectionBound::StepBound {
                steps: 1,
            }))
            .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn collect(&mut self) -> Vec<Self::B> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::Collect(self.collection_bound.clone()))
                .unwrap();
        }
        let mut buffers = vec![];
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let WorkerResult::Collected(buffer) = rx.recv().unwrap() else {
                panic!()
            };
            buffers.push(buffer);
        }
        buffers
    }
}

pub struct VecEnvWorker<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone> {
    buffer: B,
    env: E,
    policy: Option<Box<dyn Policy<Tensor = E::Tensor>>>,
    last_state: Option<E::Tensor>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone> VecEnvWorker<E, B> {
    pub fn new(buffer: B, env: E) -> Self {
        Self {
            buffer,
            env,
            policy: None,
            last_state: None,
        }
    }

    fn set_policy<P: Policy<Tensor = E::Tensor> + Clone>(&mut self, policy: P) {
        self.policy = Some(Box::new(policy))
    }

    fn get_buffer(&self) -> B {
        self.buffer.clone()
    }

    pub fn step(&mut self) {
        let Some(distr) = &mut self.policy else {
            todo!()
        };
        let last_state = self.last_state.take();
        self.buffer.step(&mut self.env, distr, last_state);
    }

    fn collect(&mut self, bound: CollectionBound) -> B {
        match bound {
            CollectionBound::StepBound { steps } => {
                for _ in 0..steps {
                    self.step();
                }
            }
            CollectionBound::EpisodeBound { steps } => {
                let mut steps_taken = 0;
                loop {
                    self.step();
                    steps_taken += 1;
                    if steps_taken >= steps && self.buffer.last_state_terminates() {
                        break;
                    }
                }
            }
        }
        self.buffer.clone()
    }
}

pub struct VecEnvPool<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone> {
    pub workers: Vec<VecEnvWorker<E, B>>,
    pub collection_bound: CollectionBound,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Send + Clone> EnvPool for VecEnvPool<E, B> {
    type E = E;
    type B = B;

    fn collection_bound(&self) -> CollectionBound {
        self.collection_bound.clone()
    }

    fn set_policy<P: Policy<Tensor = <Self::E as Env>::Tensor> + Clone>(&mut self, policy: P) {
        for worker in self.workers.iter_mut() {
            worker.set_policy(policy.clone());
        }
    }

    fn get_buffers(&self) -> Vec<Self::B> {
        self.workers.iter().map(|w| w.get_buffer()).collect()
    }

    fn collect(&mut self) -> Vec<Self::B> {
        self.workers
            .iter_mut()
            .map(|w| w.collect(self.collection_bound.clone()))
            .collect()
    }

    fn single_step(&mut self) {
        for worker in self.workers.iter_mut() {
            worker.step();
        }
    }
}
