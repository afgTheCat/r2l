use std::{fmt::Debug, sync::Arc};

use anyhow::Result;

use crate::tensor::R2lTensor;

#[derive(Debug, Clone)]
pub enum Space<T> {
    Discrete(usize),
    Continuous {
        min: Option<T>,
        max: Option<T>,
        size: usize,
    },
}

impl<T> Space<T> {
    pub fn continuous_from_dims(dims: Vec<usize>) -> Self {
        Self::Continuous {
            min: None,
            max: None,
            size: dims.iter().product(),
        }
    }

    pub fn size(&self) -> usize {
        match &self {
            Self::Discrete(size) => *size,
            Self::Continuous { size, .. } => *size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnvDescription<T> {
    pub observation_space: Space<T>,
    pub action_space: Space<T>,
}

impl<T> EnvDescription<T> {
    pub fn new(observation_space: Space<T>, action_space: Space<T>) -> Self {
        Self {
            observation_space,
            action_space,
        }
    }

    pub fn action_size(&self) -> usize {
        self.action_space.size()
    }

    pub fn observation_size(&self) -> usize {
        self.observation_space.size()
    }
}

// TODO: This is a useful thing buffer needs to be go
pub struct Snapshot<T> {
    pub state: T,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

pub trait Env {
    //  TODO: we might want to introduce more than just one kind of Tensors.
    type Tensor: R2lTensor;

    fn reset(&mut self, seed: u64) -> Result<Self::Tensor>;
    fn step(&mut self, action: Self::Tensor) -> Result<Snapshot<Self::Tensor>>;
    fn env_description(&self) -> EnvDescription<Self::Tensor>;
}

pub type EnvTensor<E> = <E as Env>::Tensor;

pub trait EnvBuilderTrait: Sync + Send + 'static {
    type Tensor: R2lTensor;
    type Env: Env<Tensor = Self::Tensor>;

    fn build_env(&self) -> Result<Self::Env>;

    fn env_description(&self) -> Result<EnvDescription<Self::Tensor>> {
        let env = self.build_env()?;
        Ok(env.env_description())
    }
}

impl<E: Env, F: Sync + Send + 'static> EnvBuilderTrait for F
where
    F: Fn() -> Result<E>,
{
    type Tensor = E::Tensor;
    type Env = E;

    fn build_env(&self) -> Result<Self::Env> {
        (self)()
    }
}

// how the environment builders should be injected to to the holder
pub enum EnvBuilder<EB: EnvBuilderTrait> {
    Homogenous { builder: Arc<EB>, n_envs: usize },
    Heterogenous { builders: Vec<Arc<EB>> },
}

impl<EB: EnvBuilderTrait> EnvBuilder<EB> {
    pub fn homogenous(builder: EB, n_envs: usize) -> Self {
        Self::Homogenous {
            builder: Arc::new(builder),
            n_envs,
        }
    }

    pub fn build_idx(&self, idx: usize) -> Result<EB::Env> {
        match &self {
            Self::Homogenous { builder, .. } => builder.build_env(),
            Self::Heterogenous { builders } => builders[idx].build_env(),
        }
    }

    pub fn num_envs(&self) -> usize {
        match self {
            Self::Homogenous { n_envs, .. } => *n_envs,
            Self::Heterogenous { builders } => builders.len(),
        }
    }

    pub fn env_builder(&self) -> Arc<EB> {
        match &self {
            Self::Homogenous { builder, .. } => builder.clone(),
            Self::Heterogenous { builders } => builders[0].clone(),
        }
    }

    pub fn env_description(&self) -> Result<EnvDescription<EB::Tensor>> {
        match &self {
            Self::Homogenous { builder, n_envs: _ } => builder.env_description(),
            Self::Heterogenous { builders } => builders[0].env_description(),
        }
    }
}
