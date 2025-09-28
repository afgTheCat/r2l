use crate::{
    env::{Env, EnvBuilderTrait, EnvironmentDescription, Memory},
    sampler::{
        env_pools::{
            thread_env_pool::{
                FixedSizeThreadEnvPool, FixedSizeWorkerCommand, FixedSizeWorkerResult,
                FixedSizeWorkerThread, VariableSizedThreadEnvPool, VariableSizedWorkerCommand,
                VariableSizedWorkerResult, VariableSizedWorkerThread,
            },
            vec_env_pool::{FixedSizeVecEnvPool, VariableSizedVecEnvPool},
        },
        trajectory_buffers::{
            fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
            variable_size_buffer::{VariableSizedStateBuffer, VariableSizedTrajectoryBuffer},
        },
    },
    sampler2::{
        Buffer, CollectionBound,
        buffers::{ArcBufferWrapper, RcBufferWrapper},
        env_pools::{ThreadEnvPool, ThreadEnvWorker, VecEnvPool, VecEnvWorker},
    },
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

// how the environment builders should be injected to to the holder
pub enum EnvBuilderType2<EB: EnvBuilderTrait> {
    EnvBuilder { builder: Arc<EB>, n_envs: usize },
    EnvBuilderVec { builders: Vec<Arc<EB>> },
}

impl<EB: EnvBuilderTrait> EnvBuilderType2<EB> {
    // TODO: should this also launch the worker threads?
    pub fn build_fixed_sized_thread(&self, capacity: usize) -> FixedSizeThreadEnvPool<EB::Env> {
        let mut channels = HashMap::new();
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                for id in 0..*n_envs {
                    let (command_tx, command_rx) =
                        crossbeam::channel::unbounded::<FixedSizeWorkerCommand<EB::Env>>();
                    let (result_tx, result_rx) =
                        crossbeam::channel::unbounded::<FixedSizeWorkerResult<EB::Env>>();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker =
                            FixedSizeWorkerThread::new(result_tx, command_rx, env, capacity);
                        worker.handle_commands();
                    });
                }
            }
            Self::EnvBuilderVec { builders } => {
                for (id, builder) in builders.iter().enumerate() {
                    let (command_tx, command_rx) =
                        crossbeam::channel::unbounded::<FixedSizeWorkerCommand<EB::Env>>();
                    let (result_tx, result_rx) =
                        crossbeam::channel::unbounded::<FixedSizeWorkerResult<EB::Env>>();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker =
                            FixedSizeWorkerThread::new(result_tx, command_rx, env, capacity);
                        worker.handle_commands();
                    });
                }
            }
        }
        FixedSizeThreadEnvPool::new(channels)
    }

    pub fn build_fixed_sized_vec(&self, capacity: usize) -> FixedSizeVecEnvPool<EB::Env> {
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                let buffers = (0..*n_envs)
                    .map(|_| FixedSizeTrajectoryBuffer::new(builder.build_env().unwrap(), capacity))
                    .collect();
                FixedSizeVecEnvPool { buffers }
            }
            Self::EnvBuilderVec { builders } => {
                let buffers = builders
                    .iter()
                    .map(|b| FixedSizeTrajectoryBuffer::new(b.build_env().unwrap(), capacity))
                    .collect();
                FixedSizeVecEnvPool { buffers }
            }
        }
    }

    pub fn build_variable_sized_vec(&self) -> VariableSizedVecEnvPool<EB::Env> {
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                let buffers = (0..*n_envs)
                    .map(|_| VariableSizedTrajectoryBuffer::new(builder.build_env().unwrap()))
                    .collect();
                VariableSizedVecEnvPool { buffers }
            }
            Self::EnvBuilderVec { builders } => {
                let buffers = builders
                    .iter()
                    .map(|b| VariableSizedTrajectoryBuffer::new(b.build_env().unwrap()))
                    .collect();
                VariableSizedVecEnvPool { buffers }
            }
        }
    }

    pub fn build_variable_sized_thread(&self) -> VariableSizedThreadEnvPool<EB::Env> {
        let mut channels = HashMap::new();
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                for id in 0..*n_envs {
                    let (command_tx, command_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerCommand<EB::Env>>();
                    let (result_tx, result_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerResult<EB::Env>>();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker = VariableSizedWorkerThread::new(result_tx, command_rx, env);
                        worker.handle_commmand();
                    });
                }
            }
            Self::EnvBuilderVec { builders } => {
                for (id, builder) in builders.iter().enumerate() {
                    let (command_tx, command_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerCommand<EB::Env>>();
                    let (result_tx, result_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerResult<EB::Env>>();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker = VariableSizedWorkerThread::new(result_tx, command_rx, env);
                        worker.handle_commmand();
                    });
                }
            }
        }
        VariableSizedThreadEnvPool::new(channels)
    }

    pub fn env_builder(&self) -> Arc<EB> {
        match &self {
            Self::EnvBuilder { builder, .. } => builder.clone(),
            Self::EnvBuilderVec { builders } => builders[0].clone(),
        }
    }

    fn to_thread_pool(
        &self,
        buffer_type: BufferType,
        collection_bound: CollectionBound,
    ) -> ThreadEnvPool<<EB as EnvBuilderTrait>::Env, ArcBufferKind<<EB as EnvBuilderTrait>::Env>>
    {
        let mut channels = HashMap::new();
        match self {
            EnvBuilderType2::EnvBuilder { builder, n_envs } => {
                for id in 0..*n_envs {
                    let (command_tx, command_rx) = crossbeam::channel::unbounded();
                    let (result_tx, result_rx) = crossbeam::channel::unbounded();
                    channels.insert(id, (command_tx, result_rx));
                    let eb_cloned = builder.clone();
                    let buffer_type = buffer_type.clone();
                    let collection_bound = collection_bound.clone();
                    std::thread::spawn(move || {
                        let buffer = buffer_type.build_arc(collection_bound);
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
                    let buffer_type = buffer_type.clone();
                    let collection_bound = collection_bound.clone();
                    std::thread::spawn(move || {
                        let buffer = buffer_type.build_arc(collection_bound);
                        let env = eb_cloned.build_env().unwrap();
                        let mut worker = ThreadEnvWorker::new(result_tx, command_rx, buffer, env);
                        worker.work();
                    });
                }
            }
        }
        ThreadEnvPool {
            collection_bound: collection_bound,
            channels,
        }
    }

    fn to_vec_pool(
        &self,
        buffer_type: BufferType,
        collection_bound: CollectionBound,
    ) -> VecEnvPool<<EB as EnvBuilderTrait>::Env, RcBufferKind<<EB as EnvBuilderTrait>::Env>> {
        let workers = match self {
            EnvBuilderType2::EnvBuilder { builder, n_envs } => {
                let mut workers = vec![];
                for _ in 0..*n_envs {
                    let buffer = buffer_type.build_rc(collection_bound.clone());
                    let env = builder.build_env().unwrap();
                    workers.push(VecEnvWorker::new(buffer, env));
                }
                workers
            }
            EnvBuilderType2::EnvBuilderVec { builders } => {
                let mut workers = vec![];
                for builder in builders {
                    let buffer = buffer_type.build_rc(collection_bound.clone());
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

    pub fn num_envs(&self) -> usize {
        match self {
            Self::EnvBuilder { n_envs, .. } => *n_envs,
            Self::EnvBuilderVec { builders } => builders.len(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum BufferType {
    FixedSize,
    VariableSized,
}

impl BufferType {
    fn build_arc<E: Env>(&self, collection_bound: CollectionBound) -> ArcBufferKind<E> {
        match self {
            Self::FixedSize => {
                let CollectionBound::StepBound { steps } = collection_bound else {
                    panic!("Cannot build a fixed size buffer with episode bound");
                };
                ArcBufferKind::FixedSize(ArcBufferWrapper::new(Arc::new(Mutex::new(
                    FixedSizeStateBuffer::new(steps),
                ))))
            }
            Self::VariableSized => ArcBufferKind::VariableSized(ArcBufferWrapper::new(Arc::new(
                Mutex::new(VariableSizedStateBuffer::default()),
            ))),
        }
    }

    fn build_rc<E: Env>(&self, collection_bound: CollectionBound) -> RcBufferKind<E> {
        match self {
            Self::FixedSize => {
                let CollectionBound::StepBound { steps } = collection_bound else {
                    panic!("Cannot build a fixed size buffer with episode bound");
                };
                RcBufferKind::FixedSize(RcBufferWrapper::new(FixedSizeStateBuffer::new(steps)))
            }
            Self::VariableSized => RcBufferKind::VariableSized(RcBufferWrapper::new(
                VariableSizedStateBuffer::default(),
            )),
        }
    }
}

pub enum ArcBufferKind<E: Env> {
    FixedSize(ArcBufferWrapper<FixedSizeStateBuffer<E>>),
    VariableSized(ArcBufferWrapper<VariableSizedStateBuffer<E>>),
}

impl<E: Env> Clone for ArcBufferKind<E> {
    fn clone(&self) -> Self {
        match self {
            Self::FixedSize(arc_buff) => Self::FixedSize(arc_buff.clone()),
            Self::VariableSized(arc_buff) => Self::VariableSized(arc_buff.clone()),
        }
    }
}

impl<E: Env> Buffer for ArcBufferKind<E> {
    type Tensor = <E as Env>::Tensor;

    fn states(&self) -> Vec<Self::Tensor> {
        todo!()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        todo!()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        todo!()
    }

    fn rewards(&self) -> Vec<f32> {
        todo!()
    }

    fn terminated(&self) -> Vec<bool> {
        todo!()
    }

    fn trancuated(&self) -> Vec<bool> {
        todo!()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.push(snapshot),
            Self::VariableSized(rc_buf) => rc_buf.push(snapshot),
        }
    }

    fn last_state_terminates(&self) -> bool {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.last_state_terminates(),
            Self::VariableSized(rc_buf) => rc_buf.last_state_terminates(),
        }
    }

    fn build(collection_bound: CollectionBound) -> Self {
        todo!()
    }

    fn last_state(&self) -> Option<Self::Tensor> {
        match self {
            Self::FixedSize(buff) => buff.last_state(),
            Self::VariableSized(buff) => buff.last_state(),
        }
    }
}

pub enum RcBufferKind<E: Env> {
    FixedSize(RcBufferWrapper<FixedSizeStateBuffer<E>>),
    VariableSized(RcBufferWrapper<VariableSizedStateBuffer<E>>),
}

impl<E: Env> Clone for RcBufferKind<E> {
    fn clone(&self) -> Self {
        match self {
            Self::FixedSize(rc_buf) => Self::FixedSize(rc_buf.clone()),
            Self::VariableSized(rc_buf) => Self::VariableSized(rc_buf.clone()),
        }
    }
}

impl<E: Env> Buffer for RcBufferKind<E> {
    type Tensor = <E as Env>::Tensor;

    fn states(&self) -> Vec<Self::Tensor> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.states(),
            Self::VariableSized(rc_buf) => rc_buf.states(),
        }
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.next_states(),
            Self::VariableSized(rc_buf) => rc_buf.next_states(),
        }
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.actions(),
            Self::VariableSized(rc_buf) => rc_buf.actions(),
        }
    }

    fn rewards(&self) -> Vec<f32> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.rewards(),
            Self::VariableSized(rc_buf) => rc_buf.rewards(),
        }
    }

    fn terminated(&self) -> Vec<bool> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.terminated(),
            Self::VariableSized(rc_buf) => rc_buf.terminated(),
        }
    }

    fn trancuated(&self) -> Vec<bool> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.trancuated(),
            Self::VariableSized(rc_buf) => rc_buf.trancuated(),
        }
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.push(snapshot),
            Self::VariableSized(rc_buf) => rc_buf.push(snapshot),
        }
    }

    fn last_state_terminates(&self) -> bool {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.last_state_terminates(),
            Self::VariableSized(rc_buf) => rc_buf.last_state_terminates(),
        }
    }

    fn build(collection_bound: CollectionBound) -> Self {
        todo!()
    }

    fn last_state(&self) -> Option<Self::Tensor> {
        match self {
            Self::FixedSize(buff) => buff.last_state(),
            Self::VariableSized(buff) => buff.last_state(),
        }
    }
}

pub enum BufferKind<E: Env> {
    RcBuffer(RcBufferKind<E>),
    ArcBufferKind(ArcBufferKind<E>),
}

impl<E: Env> Buffer for BufferKind<E> {
    type Tensor = <E as Env>::Tensor;

    fn states(&self) -> Vec<Self::Tensor> {
        match self {
            Self::RcBuffer(buff) => buff.states(),
            Self::ArcBufferKind(buff) => buff.states(),
        }
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        match self {
            Self::RcBuffer(buff) => buff.next_states(),
            Self::ArcBufferKind(buff) => buff.next_states(),
        }
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        match self {
            Self::RcBuffer(buff) => buff.actions(),
            Self::ArcBufferKind(buff) => buff.actions(),
        }
    }

    fn rewards(&self) -> Vec<f32> {
        match self {
            Self::RcBuffer(buff) => buff.rewards(),
            Self::ArcBufferKind(buff) => buff.rewards(),
        }
    }

    fn terminated(&self) -> Vec<bool> {
        match self {
            Self::RcBuffer(buff) => buff.terminated(),
            Self::ArcBufferKind(buff) => buff.terminated(),
        }
    }

    fn trancuated(&self) -> Vec<bool> {
        match self {
            Self::RcBuffer(buff) => buff.trancuated(),
            Self::ArcBufferKind(buff) => buff.trancuated(),
        }
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        match self {
            Self::RcBuffer(buff) => buff.push(snapshot),
            Self::ArcBufferKind(buff) => buff.push(snapshot),
        }
    }

    fn last_state_terminates(&self) -> bool {
        match self {
            Self::RcBuffer(buff) => buff.last_state_terminates(),
            Self::ArcBufferKind(buff) => buff.last_state_terminates(),
        }
    }

    fn build(collection_bound: CollectionBound) -> Self {
        todo!()
    }

    fn last_state(&self) -> Option<Self::Tensor> {
        match self {
            Self::RcBuffer(buff) => buff.last_state(),
            Self::ArcBufferKind(buff) => buff.last_state(),
        }
    }
}

// The idea is that this should be a trait?
pub enum EnvPoolType<E: Env> {
    Vec(VecEnvPool<E, RcBufferKind<E>>),
    Thread(ThreadEnvPool<E, ArcBufferKind<E>>),
}

impl<E: Env> EnvPoolType<E> {
    pub fn single_step(&mut self) {
        match self {
            Self::Vec(env_pool) => env_pool.single_step(),
            Self::Thread(env_pool) => todo!(),
        }
    }

    pub fn collection_bound(&self) -> CollectionBound {
        match self {
            Self::Vec(env_pool) => env_pool.collection_bound.clone(),
            Self::Thread(env_pool) => todo!(),
        }
    }

    pub fn set_policy<P: crate::distributions::Policy<Tensor = E::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) {
        match self {
            Self::Vec(env_pool) => env_pool.set_policy(policy),
            Self::Thread(env_pool) => todo!(),
        }
    }

    pub fn get_buffers(&self) -> Vec<BufferKind<E>> {
        match self {
            Self::Vec(env_pool) => env_pool
                .get_buffers()
                .into_iter()
                .map(|b| BufferKind::RcBuffer(b))
                .collect(),
            Self::Thread(env_pool) => todo!(),
        }
    }

    pub fn collect(&mut self) -> Vec<BufferKind<E>> {
        match self {
            Self::Vec(env_pool) => env_pool
                .collect()
                .into_iter()
                .map(|b| BufferKind::RcBuffer(b))
                .collect(),
            Self::Thread(env_pool) => todo!(),
        }
    }

    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        match &self {
            Self::Vec(vec_env) => vec_env.environment_description(),
            _ => todo!(),
        }
    }
}

pub enum WorkerLocation {
    Vec,
    Thread,
}

pub struct EnvPoolBuilder {
    // whether we want the workers to have their own threads or not
    pub worker_location: WorkerLocation,
    // collection bound
    pub collection_bound: CollectionBound,
    // Fixed size or variable sized
    pub buffer_type: BufferType,
}

// TODO: do we really need this?
impl Default for EnvPoolBuilder {
    fn default() -> Self {
        Self {
            worker_location: WorkerLocation::Vec,
            collection_bound: CollectionBound::StepBound { steps: 1024 },
            buffer_type: BufferType::FixedSize,
        }
    }
}

impl EnvPoolBuilder {
    pub fn build<EB: EnvBuilderTrait>(
        &self,
        builder: EnvBuilderType2<EB>,
    ) -> EnvPoolType<<EB as EnvBuilderTrait>::Env> {
        match self.worker_location {
            WorkerLocation::Vec => {
                let env_pool =
                    builder.to_vec_pool(self.buffer_type.clone(), self.collection_bound.clone());
                EnvPoolType::Vec(env_pool)
            }
            WorkerLocation::Thread => {
                let env_pool =
                    builder.to_thread_pool(self.buffer_type.clone(), self.collection_bound.clone());
                EnvPoolType::Thread(env_pool)
            }
        }
    }
}
