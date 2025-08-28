use crate::builders::{env::EnvBuilderTrait, sampler_hooks::SequentialEnvHookTypes};
use candle_core::Device;
use r2l_core::{
    env::Env,
    numeric::Buffer,
    sampler::{
        env_pools::{
            FixedSizeEnvPoolKind, VariableSizedEnvPoolKind,
            thread_env_pool::{
                FixedSizeThreadEnvPool, FixedSizeWorkerCommand, FixedSizeWorkerResult,
                FixedSizeWorkerThread, VariableSizedThreadEnvPool, VariableSizedWorkerCommand,
                VariableSizedWorkerResult, VariableSizedWorkerThread,
            },
            vec_env_pool::{FixedSizeVecEnvPool, VariableSizedVecEnvPool},
        },
        samplers::{CollectionType, NewSampler},
        trajectory_buffers::{
            fixed_size_buffer::FixedSizeTrajectoryBuffer,
            variable_size_buffer::VariableSizedTrajectoryBuffer,
        },
    },
};
use std::{collections::HashMap, sync::Arc};

pub enum BuilderType<EB: EnvBuilderTrait> {
    EnvBuilder { builder: Arc<EB>, n_envs: usize },
    EnvBuilderVec { builders: Vec<Arc<EB>> },
}

impl<EB: EnvBuilderTrait> BuilderType<EB> {
    fn build_fixed_sized_vec(
        &self,
        capacity: usize,
        device: &Device,
    ) -> FixedSizeVecEnvPool<EB::Env> {
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                let buffers = (0..*n_envs)
                    .map(|_| {
                        FixedSizeTrajectoryBuffer::new(builder.build_env(device).unwrap(), capacity)
                    })
                    .collect();
                FixedSizeVecEnvPool { buffers }
            }
            Self::EnvBuilderVec { builders } => {
                let buffers = builders
                    .iter()
                    .map(|b| FixedSizeTrajectoryBuffer::new(b.build_env(device).unwrap(), capacity))
                    .collect();
                FixedSizeVecEnvPool { buffers }
            }
        }
    }

    fn build_variable_sized_vec(&self, device: &Device) -> VariableSizedVecEnvPool<EB::Env> {
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                let buffers = (0..*n_envs)
                    .map(|_| VariableSizedTrajectoryBuffer::new(builder.build_env(device).unwrap()))
                    .collect();
                VariableSizedVecEnvPool { buffers }
            }
            Self::EnvBuilderVec { builders } => {
                let buffers = builders
                    .iter()
                    .map(|b| VariableSizedTrajectoryBuffer::new(b.build_env(device).unwrap()))
                    .collect();
                VariableSizedVecEnvPool { buffers }
            }
        }
    }

    // TODO: should this also launch the worker threads?
    fn build_fixed_sized_thread(
        &self,
        capacity: usize,
        device: &Device,
    ) -> FixedSizeThreadEnvPool<EB::Env> {
        let mut channels = HashMap::new();
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                for id in 0..*n_envs {
                    let (command_tx, command_rx) =
                        crossbeam::channel::unbounded::<FixedSizeWorkerCommand<EB::Env>>();
                    let (result_tx, result_rx) =
                        crossbeam::channel::unbounded::<FixedSizeWorkerResult<EB::Env>>();
                    channels.insert(id, (command_tx, result_rx));
                    let device_cloned = device.clone();
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env(&device_cloned).unwrap();
                        let mut worker =
                            FixedSizeWorkerThread::new(result_tx, command_rx, env, capacity);
                        worker.handle_commmands();
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
                    let device_cloned = device.clone();
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env(&device_cloned).unwrap();
                        let mut worker =
                            FixedSizeWorkerThread::new(result_tx, command_rx, env, capacity);
                        worker.handle_commmands();
                    });
                }
            }
        }
        FixedSizeThreadEnvPool::new(channels)
    }

    fn build_variable_sized_thread(&self, device: &Device) -> VariableSizedThreadEnvPool<EB::Env> {
        let mut channels = HashMap::new();
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                for id in 0..*n_envs {
                    let (command_tx, command_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerCommand>();
                    let (result_tx, result_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerResult>();
                    channels.insert(id, (command_tx, result_rx));
                    let device_cloned = device.clone();
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env(&device_cloned).unwrap();
                        let mut worker = VariableSizedWorkerThread::new(result_tx, command_rx, env);
                        worker.handle_commmand();
                    });
                }
            }
            Self::EnvBuilderVec { builders } => {
                for (id, builder) in builders.iter().enumerate() {
                    let (command_tx, command_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerCommand>();
                    let (result_tx, result_rx) =
                        crossbeam::channel::unbounded::<VariableSizedWorkerResult>();
                    channels.insert(id, (command_tx, result_rx));
                    let device_cloned = device.clone();
                    let eb_cloned = builder.clone();
                    std::thread::spawn(move || {
                        let env = eb_cloned.build_env(&device_cloned).unwrap();
                        let mut worker = VariableSizedWorkerThread::new(result_tx, command_rx, env);
                        worker.handle_commmand();
                    });
                }
            }
        }
        VariableSizedThreadEnvPool::new(channels)
    }
}

pub enum EnvPoolType {
    VecVariable,
    VecStep,
    ThreadVariable,
    ThreadStep,
}

pub struct SamplerType {
    capacity: usize,
    hook: SequentialEnvHookTypes,
    env_pool_type: EnvPoolType,
}

impl SamplerType {
    pub fn build<E: Env<Tensor = Buffer>, EB: EnvBuilderTrait<Env = E>>(
        &self,
        builder_type: BuilderType<EB>,
        device: &Device,
    ) -> NewSampler<E> {
        let collection_type = match self.env_pool_type {
            EnvPoolType::VecVariable => {
                let env_pool = builder_type.build_variable_sized_vec(device);
                let env_pool = VariableSizedEnvPoolKind::VariableSizedVecEnvPool(env_pool);
                CollectionType::EpisodeBound { env_pool }
            }
            EnvPoolType::VecStep => {
                let hooks = self.hook.build();
                let env_pool = builder_type.build_fixed_sized_vec(self.capacity, device);
                let env_pool = FixedSizeEnvPoolKind::FixedSizeVecEnvPool(env_pool);
                CollectionType::StepBound { env_pool, hooks }
            }
            EnvPoolType::ThreadVariable => {
                let env_pool = builder_type.build_variable_sized_thread(device);
                let env_pool = VariableSizedEnvPoolKind::VariableSizedThreadEnvPool(env_pool);
                CollectionType::EpisodeBound { env_pool }
            }
            EnvPoolType::ThreadStep => {
                let hooks = self.hook.build();
                let env_pool = builder_type.build_fixed_sized_thread(self.capacity, device);
                let env_pool = FixedSizeEnvPoolKind::FixedSizeThreadEnvPool(env_pool);
                CollectionType::StepBound { env_pool, hooks }
            }
        };
        NewSampler {
            env_steps: self.capacity,
            collection_type,
        }
    }
}
