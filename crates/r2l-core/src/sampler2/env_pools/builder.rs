use crate::{
    env::{Env, EnvBuilderTrait, Memory},
    sampler::trajectory_buffers::{
        fixed_size_buffer::FixedSizeStateBuffer, variable_size_buffer::VariableSizedStateBuffer,
    },
    sampler2::{
        Buffer, CollectionBound,
        buffers::{ArcBufferWrapper, RcBufferWrapper},
        env_pools::{ThreadEnvPool, ThreadEnvWorker, VecEnvPool, VecEnvWorker},
    },
};
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{Arc, Mutex},
};

// how the environment builders should be injected to to the holder
pub enum EnvBuilderType<EB: EnvBuilderTrait> {
    EnvBuilder { builder: Arc<EB>, n_envs: usize },
    EnvBuilderVec { builders: Vec<Arc<EB>> },
}

impl<EB: EnvBuilderTrait> EnvBuilderType<EB> {
    fn to_thread_pool(
        &self,
        buffer_type: BufferType,
        collection_bound: CollectionBound,
    ) -> ThreadEnvPool<<EB as EnvBuilderTrait>::Env, ArcBufferKind<<EB as EnvBuilderTrait>::Env>>
    {
        let mut channels = HashMap::new();
        match self {
            EnvBuilderType::EnvBuilder { builder, n_envs } => {
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
            EnvBuilderType::EnvBuilderVec { builders } => {
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
            EnvBuilderType::EnvBuilder { builder, n_envs } => {
                let mut workers = vec![];
                for _ in 0..*n_envs {
                    let buffer = buffer_type.build_rc(collection_bound.clone());
                    let env = builder.build_env().unwrap();
                    workers.push(VecEnvWorker::new(buffer, env));
                }
                workers
            }
            EnvBuilderType::EnvBuilderVec { builders } => {
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
                RcBufferKind::FixedSize(RcBufferWrapper::new(Rc::new(RefCell::new(
                    FixedSizeStateBuffer::new(steps),
                ))))
            }
            Self::VariableSized => RcBufferKind::VariableSized(RcBufferWrapper::new(Rc::new(
                RefCell::new(VariableSizedStateBuffer::default()),
            ))),
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

    fn last_state(&self) -> Option<Self::Tensor> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.last_state(),
            Self::VariableSized(rc_buf) => rc_buf.last_state(),
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

    fn last_state(&self) -> Option<Self::Tensor> {
        match self {
            Self::FixedSize(rc_buf) => rc_buf.last_state(),
            Self::VariableSized(rc_buf) => rc_buf.last_state(),
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
}

pub enum EnvPoolType<E: Env> {
    Vec(VecEnvPool<E, RcBufferKind<E>>),
    Thread(ThreadEnvPool<E, ArcBufferKind<E>>),
}

pub enum WorkerLocation {
    Vec,
    Thread,
}

pub struct EnvPoolBuilder<EB: EnvBuilderTrait> {
    // whether we want the workers to have their own threads or not
    worker_location: WorkerLocation,
    // collection bound
    collection_bound: CollectionBound,
    // environment builders
    builder: EnvBuilderType<EB>,
    // Fixed size or variable sized
    buffer_type: BufferType,
}

impl<EB: EnvBuilderTrait> EnvPoolBuilder<EB> {
    pub fn build(&self) -> EnvPoolType<<EB as EnvBuilderTrait>::Env> {
        match self.worker_location {
            WorkerLocation::Vec => {
                let env_pool = self
                    .builder
                    .to_vec_pool(self.buffer_type.clone(), self.collection_bound.clone());
                EnvPoolType::Vec(env_pool)
            }
            WorkerLocation::Thread => {
                let env_pool = self
                    .builder
                    .to_thread_pool(self.buffer_type.clone(), self.collection_bound.clone());
                EnvPoolType::Thread(env_pool)
            }
        }
    }
}
