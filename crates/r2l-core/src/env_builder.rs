use crate::{
    env::{Env, EnvironmentDescription},
    tensor::R2lTensor,
};
use anyhow::Result;
use std::sync::Arc;

pub trait EnvBuilderTrait: Sync + Send + 'static {
    type Tensor: R2lTensor;
    type Env: Env<Tensor = Self::Tensor>;

    fn build_env(&self) -> Result<Self::Env>;

    fn env_description(&self) -> Result<EnvironmentDescription<Self::Tensor>> {
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

    pub fn env_description(&self) -> Result<EnvironmentDescription<EB::Tensor>> {
        match &self {
            Self::Homogenous { builder, n_envs: _ } => builder.env_description(),
            Self::Heterogenous { builders } => builders[0].env_description(),
        }
    }
}
