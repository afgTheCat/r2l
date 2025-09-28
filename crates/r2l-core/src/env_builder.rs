use crate::{
    env::Env,
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
            fixed_size_buffer::FixedSizeTrajectoryBuffer,
            variable_size_buffer::VariableSizedTrajectoryBuffer,
        },
    },
};
use anyhow::Result;
use std::{collections::HashMap, sync::Arc};

pub trait EnvBuilderTrait: Sync + Send + 'static {
    type Env: Env;

    fn build_env(&self) -> Result<Self::Env>;
}

impl<E: Env, F: Sync + Send + 'static> EnvBuilderTrait for F
where
    F: Fn() -> Result<E>,
{
    type Env = E;

    fn build_env(&self) -> Result<Self::Env> {
        (self)()
    }
}

// how the environment builders should be injected to to the holder
pub enum EnvBuilderType<EB: EnvBuilderTrait> {
    EnvBuilder { builder: Arc<EB>, n_envs: usize },
    EnvBuilderVec { builders: Vec<Arc<EB>> },
}

impl<EB: EnvBuilderTrait> EnvBuilderType<EB> {
    pub fn num_envs(&self) -> usize {
        match self {
            Self::EnvBuilder { n_envs, .. } => *n_envs,
            Self::EnvBuilderVec { builders } => builders.len(),
        }
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
}
