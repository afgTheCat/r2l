pub mod builder;

use crate::{
    distributions::Policy,
    env::{Env, EnvBuilderTrait, EnvironmentDescription},
    sampler2::{
        Buffer, CollectionBound, buffers::RcBufferWrapper, env_pools::builder::EnvBuilderType2,
    },
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
    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
    ) -> Self {
        todo!()
    }
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

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + 'static> ThreadEnvPool<E, B> {
    fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
    ) -> Self
    where
        B: Clone + Send,
        E: 'static,
    {
        let mut channels = HashMap::new();
        match env_builder {
            EnvBuilderType2::EnvBuilder { builder, n_envs } => {
                for id in 0..n_envs {
                    let (command_tx, command_rx) = crossbeam::channel::unbounded();
                    let (result_tx, result_rx) = crossbeam::channel::unbounded();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    let collection_bound = collection_bound.clone();
                    std::thread::spawn(move || {
                        let buffer = B::build(collection_bound);
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker = ThreadEnvWorker::new(result_tx, command_rx, buffer, env);
                        worker.work();
                    });
                }
            }
            EnvBuilderType2::EnvBuilderVec { builders } => {
                for (id, builder) in builders.iter().enumerate() {
                    let (command_tx, command_rx) = crossbeam::channel::unbounded();
                    let (result_tx, result_rx) = crossbeam::channel::unbounded();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    let collection_bound = collection_bound.clone();
                    std::thread::spawn(move || {
                        let buffer = B::build(collection_bound);
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker = ThreadEnvWorker::new(result_tx, command_rx, buffer, env);
                        worker.work();
                    });
                }
            }
        }
        Self {
            channels,
            collection_bound,
        }
    }

    pub fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn collection_bound(&self) -> CollectionBound {
        self.collection_bound.clone()
    }

    fn set_policy<P: Policy<Tensor = E::Tensor> + Clone>(&mut self, policy: P) {
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

    fn get_buffers(&self) -> Vec<B> {
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

    fn collect(&mut self) -> Vec<B> {
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

    pub fn environment_description(&self) -> EnvironmentDescription<E::Tensor> {
        self.env.env_description()
    }
}

pub struct VecEnvPool<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone> {
    pub workers: Vec<VecEnvWorker<E, B>>,
    pub collection_bound: CollectionBound,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone> VecEnvPool<E, B> {
    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
    ) -> Self {
        let workers = match env_builder {
            EnvBuilderType2::EnvBuilder { builder, n_envs } => {
                let mut workers = vec![];
                for _ in 0..n_envs {
                    let buffer = B::build(collection_bound.clone());
                    let env = builder.build_env().unwrap();
                    workers.push(VecEnvWorker::new(buffer, env));
                }
                workers
            }
            EnvBuilderType2::EnvBuilderVec { builders } => {
                let mut workers = vec![];
                for builder in builders {
                    let buffer = B::build(collection_bound.clone());
                    let env = builder.build_env().unwrap();
                    workers.push(VecEnvWorker::new(buffer, env));
                }
                workers
            }
        };
        VecEnvPool {
            workers,
            collection_bound,
        }
    }
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone> VecEnvPool<E, B> {
    fn environment_description(&self) -> EnvironmentDescription<E::Tensor> {
        self.workers[0].environment_description()
    }

    fn collection_bound(&self) -> CollectionBound {
        self.collection_bound.clone()
    }

    fn set_policy<P: Policy<Tensor = E::Tensor> + Clone>(&mut self, policy: P) {
        for worker in self.workers.iter_mut() {
            worker.set_policy(policy.clone());
        }
    }

    fn get_buffers(&self) -> Vec<B> {
        self.workers.iter().map(|w| w.get_buffer()).collect()
    }

    fn collect(&mut self) -> Vec<B> {
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
