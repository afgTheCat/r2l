use crate::env::Env;
use anyhow::Result;
use std::sync::Arc;

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
    pub fn build_idx(&self, idx: usize) -> Result<EB::Env> {
        match &self {
            Self::EnvBuilder { builder, .. } => builder.build_env(),
            Self::EnvBuilderVec { builders } => builders[idx].build_env(),
        }
    }

    pub fn num_envs(&self) -> usize {
        match self {
            Self::EnvBuilder { n_envs, .. } => *n_envs,
            Self::EnvBuilderVec { builders } => builders.len(),
        }
    }

    pub fn env_builder(&self) -> Arc<EB> {
        match &self {
            Self::EnvBuilder { builder, .. } => builder.clone(),
            Self::EnvBuilderVec { builders } => builders[0].clone(),
        }
    }
}
