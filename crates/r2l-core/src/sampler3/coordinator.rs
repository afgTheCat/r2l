use crate::{
    distributions::Policy,
    env::{Env, EnvBuilderTrait, EnvironmentDescription},
    sampler2::{Buffer, CollectionBound, env_pools::builder::EnvBuilderType2},
    sampler3::buffers::{ArcBufferWrapper, BufferStack, RcBufferWrapper},
    tensor::R2lTensor,
};
use crossbeam::channel::{Receiver, Sender};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub struct VecEnvWorker<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> {
    buffer: RcBufferWrapper<B>,
    env: E,
    policy: Option<Box<dyn Policy<Tensor = E::Tensor>>>,
    last_state: Option<E::Tensor>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> VecEnvWorker<E, B> {
    pub fn new(buffer: RcBufferWrapper<B>, env: E) -> Self {
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

    pub fn step(&mut self) {
        let Some(distr) = &mut self.policy else {
            todo!()
        };
        let last_state = self.last_state.take();
        self.buffer
            .buffer_mut()
            .step(&mut self.env, distr, last_state);
    }

    fn collect(&mut self, bound: CollectionBound) {
        let Some(distr) = &mut self.policy else {
            todo!()
        };
        let mut buffer = self.buffer.buffer_mut();
        match bound {
            CollectionBound::StepBound { steps } => {
                for _ in 0..steps {
                    let last_state = self.last_state.take();
                    buffer.step(&mut self.env, distr, last_state);
                }
            }
            CollectionBound::EpisodeBound { steps } => {
                let mut steps_taken = 0;
                loop {
                    let last_state = self.last_state.take();
                    buffer.step(&mut self.env, distr, last_state);
                    steps_taken += 1;
                    if steps_taken >= steps && buffer.last_state_terminates() {
                        break;
                    }
                }
            }
        }
    }

    pub fn environment_description(&self) -> EnvironmentDescription<E::Tensor> {
        self.env.env_description()
    }
}

pub enum WorkerCommand2<T: R2lTensor> {
    SetPolicy(Box<dyn Policy<Tensor = T>>),
    Collect(CollectionBound),
}

pub enum WorkerResult2 {
    PolicySet,
    Collected,
}

pub type ChannelHashMap<T> = HashMap<usize, (Sender<WorkerCommand2<T>>, Receiver<WorkerResult2>)>;

enum CoordinatorType<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> {
    Vec(Vec<VecEnvWorker<E, B>>),
    ThreadEnvWorker {
        channels: ChannelHashMap<E::Tensor>,
        buffers: Vec<ArcBufferWrapper<B>>,
    },
}

pub enum Location {
    Vec,
    Thread,
}
struct ThreadEnvWorker2<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> {
    tx: Sender<WorkerResult2>,
    rx: Receiver<WorkerCommand2<<E as Env>::Tensor>>,
    buffer: ArcBufferWrapper<B>,
    env: E,
    policy: Option<Box<dyn Policy<Tensor = E::Tensor>>>,
    last_state: Option<E::Tensor>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> ThreadEnvWorker2<E, B> {
    fn new(
        tx: Sender<WorkerResult2>,
        rx: Receiver<WorkerCommand2<<E as Env>::Tensor>>,
        buffer: ArcBufferWrapper<B>,
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
        self.buffer.buffer().step(&mut self.env, distr, last_state);
    }

    pub fn collect(&mut self, bound: CollectionBound) {
        let Some(distr) = &mut self.policy else {
            todo!()
        };
        let mut buffer = self.buffer.buffer();
        match bound {
            CollectionBound::StepBound { steps } => {
                for _ in 0..steps {
                    let last_state = self.last_state.take();
                    buffer.step(&mut self.env, distr, last_state);
                }
            }
            CollectionBound::EpisodeBound { steps } => {
                let mut steps_taken = 0;
                loop {
                    let last_state = self.last_state.take();
                    buffer.step(&mut self.env, distr, last_state);
                    steps_taken += 1;
                    if steps_taken >= steps && buffer.last_state_terminates() {
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
                WorkerCommand2::SetPolicy(distr) => {
                    self.policy = Some(distr);
                    self.tx.send(WorkerResult2::PolicySet).unwrap();
                }
                WorkerCommand2::Collect(bound) => {
                    self.collect(bound);
                    self.tx.send(WorkerResult2::Collected).unwrap();
                }
            }
        }
    }
}

pub struct CoordinatorS<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> {
    collection_bound: CollectionBound,
    coordinator_type: CoordinatorType<E, B>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> CoordinatorS<E, B> {
    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        match &self.coordinator_type {
            CoordinatorType::ThreadEnvWorker { channels, buffers } => {
                todo!()
            }
            CoordinatorType::Vec(workers) => workers[0].environment_description(),
        }
    }

    pub fn set_policy<P: crate::distributions::Policy<Tensor = E::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) {
        match &mut self.coordinator_type {
            CoordinatorType::ThreadEnvWorker { channels, .. } => {
                let num_envs = channels.len();
                for idx in 0..num_envs {
                    let tx = &channels.get(&idx).unwrap().0;
                    tx.send(WorkerCommand2::SetPolicy(Box::new(policy.clone())))
                        .unwrap();
                }
                for idx in 0..num_envs {
                    let rx = &channels.get(&idx).unwrap().1;
                    rx.recv().unwrap();
                }
            }
            CoordinatorType::Vec(workers) => {
                for worker in workers.iter_mut() {
                    // TODO: we need to ensure deep copy here once SDE sampling is not an issue
                    worker.policy = Some(Box::new(policy.clone()));
                }
            }
        }
    }

    pub fn collection_bound(&self) -> CollectionBound {
        self.collection_bound.clone()
    }

    pub fn get_buffers(&self) -> BufferStack<B> {
        match &self.coordinator_type {
            CoordinatorType::ThreadEnvWorker { buffers, .. } => {
                BufferStack::AtomicRefCounted(buffers.iter().cloned().collect::<Vec<_>>())
            }
            CoordinatorType::Vec(workers) => BufferStack::RefCounted(
                workers.iter().map(|w| w.buffer.clone()).collect::<Vec<_>>(),
            ),
        }
    }

    pub fn single_step(&mut self) {
        match &mut self.coordinator_type {
            CoordinatorType::ThreadEnvWorker { channels, .. } => {
                let num_envs = channels.len();
                for idx in 0..num_envs {
                    let tx = &channels.get(&idx).unwrap().0;
                    tx.send(WorkerCommand2::Collect(CollectionBound::StepBound {
                        steps: 1,
                    }))
                    .unwrap();
                }
                for idx in 0..num_envs {
                    let rx = &channels.get(&idx).unwrap().1;
                    rx.recv().unwrap();
                }
            }
            CoordinatorType::Vec(workers) => {
                for worker in workers.iter_mut() {
                    worker.step();
                }
            }
        }
    }

    pub fn collect(&mut self) {
        match &mut self.coordinator_type {
            CoordinatorType::ThreadEnvWorker { channels, .. } => {
                let num_envs = channels.len();
                for idx in 0..num_envs {
                    let tx = &channels.get(&idx).unwrap().0;
                    tx.send(WorkerCommand2::Collect(self.collection_bound.clone()))
                        .unwrap();
                }
                for idx in 0..num_envs {
                    let rx = &channels.get(&idx).unwrap().1;
                    rx.recv().unwrap();
                }
            }
            CoordinatorType::Vec(workers) => {
                // NOTE: OLD WAY
                for worker in workers.iter_mut() {
                    worker.collect(self.collection_bound.clone());
                }
            }
        }
    }
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> CoordinatorS<E, B> {
    pub fn build_rc<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
    ) -> Self {
        let workers = match env_builder {
            EnvBuilderType2::EnvBuilder { builder, n_envs } => {
                let mut workers = vec![];
                for _ in 0..n_envs {
                    let buffer = B::build(collection_bound.clone());
                    let buffer = RcBufferWrapper::new(buffer);
                    let env = builder.build_env().unwrap();
                    workers.push(VecEnvWorker::new(buffer, env));
                }
                workers
            }
            EnvBuilderType2::EnvBuilderVec { builders } => {
                let mut workers = vec![];
                for builder in builders {
                    let buffer = B::build(collection_bound.clone());
                    let buffer = RcBufferWrapper::new(buffer);
                    let env = builder.build_env().unwrap();
                    workers.push(VecEnvWorker::new(buffer, env));
                }
                workers
            }
        };
        Self {
            coordinator_type: CoordinatorType::Vec(workers),
            collection_bound,
        }
    }
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Send + 'static> CoordinatorS<E, B> {
    pub fn build_arc<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
    ) -> Self {
        let mut channels = HashMap::new();
        let mut buffers = vec![];
        match env_builder {
            EnvBuilderType2::EnvBuilder { builder, n_envs } => {
                for id in 0..n_envs {
                    let (command_tx, command_rx) = crossbeam::channel::unbounded();
                    let (result_tx, result_rx) = crossbeam::channel::unbounded();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    let collection_bound = collection_bound.clone();
                    let buffer = ArcBufferWrapper(Arc::new(Mutex::new(B::build(collection_bound))));
                    let buffer_c = buffer.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker =
                            ThreadEnvWorker2::new(result_tx, command_rx, buffer_c, env);
                        worker.work();
                    });
                    buffers.push(buffer)
                }
            }
            EnvBuilderType2::EnvBuilderVec { builders } => {
                for (id, builder) in builders.iter().enumerate() {
                    let (command_tx, command_rx) = crossbeam::channel::unbounded();
                    let (result_tx, result_rx) = crossbeam::channel::unbounded();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    let collection_bound = collection_bound.clone();
                    let buffer = ArcBufferWrapper(Arc::new(Mutex::new(B::build(collection_bound))));
                    let buffer_c = buffer.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker =
                            ThreadEnvWorker2::new(result_tx, command_rx, buffer_c, env);
                        worker.work();
                    });
                    buffers.push(buffer);
                }
            }
        }
        Self {
            coordinator_type: CoordinatorType::ThreadEnvWorker {
                channels,
                buffers: vec![],
            },
            collection_bound,
        }
    }
}
