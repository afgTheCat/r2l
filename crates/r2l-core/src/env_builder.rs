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
}
