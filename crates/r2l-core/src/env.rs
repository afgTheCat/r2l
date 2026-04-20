use std::{fmt::Debug, sync::Arc};

use anyhow::Result;

use crate::tensor::R2lTensor;

/// The type of action space.
#[derive(Debug, Clone, Copy)]
pub enum ActionSpaceType {
    Discrete,
    Continuous,
}

/// Description of an observation or action space.
#[derive(Debug, Clone)]
pub enum Space<T> {
    /// Discrete space with `usize` possible values.
    Discrete(usize),
    /// Continuous vector space with optional elementwise bounds.
    Continuous {
        /// Optional minimum values.
        min: Option<T>,
        /// Optional maximum values.
        max: Option<T>,
        /// Number of scalar elements in the flattened space.
        size: usize,
    },
}

impl<T> Space<T> {
    /// Creates an unbounded continuous space from tensor dimensions.
    pub fn continuous_from_dims(dims: Vec<usize>) -> Self {
        Self::Continuous {
            min: None,
            max: None,
            size: dims.iter().product(),
        }
    }

    /// Returns the flattened size of the space.
    pub fn size(&self) -> usize {
        match &self {
            Self::Discrete(size) => *size,
            Self::Continuous { size, .. } => *size,
        }
    }
}

/// Observation and action space metadata for an environment.
#[derive(Debug, Clone)]
pub struct EnvDescription<T> {
    /// Space returned by [`Env::reset`] and [`Env::step`].
    pub observation_space: Space<T>,
    /// Space accepted by [`Env::step`].
    pub action_space: Space<T>,
}

impl<T> EnvDescription<T> {
    /// Creates a description from observation and action spaces.
    pub fn new(observation_space: Space<T>, action_space: Space<T>) -> Self {
        Self {
            observation_space,
            action_space,
        }
    }

    /// Returns the flattened action-space size.
    pub fn action_size(&self) -> usize {
        self.action_space.size()
    }

    /// Returns the flattened observation-space size.
    pub fn observation_size(&self) -> usize {
        self.observation_space.size()
    }
}

/// Result of one environment step.
pub struct Snapshot<T> {
    /// Observation after the action was applied.
    pub state: T,
    /// Reward produced by the transition.
    pub reward: f32,
    /// Whether the environment reached a terminal state.
    pub terminated: bool,
    /// Whether the episode ended because of a time limit or external cutoff.
    pub truncated: bool,
}

// ANCHOR: env
/// Environment interface used by samplers.
pub trait Env {
    /// Tensor type used for observations and actions.
    type Tensor: R2lTensor;

    /// Resets the environment and returns the initial observation.
    fn reset(&mut self, seed: u64) -> Result<Self::Tensor>;
    /// Applies one action and returns the resulting transition snapshot.
    fn step(&mut self, action: Self::Tensor) -> Result<Snapshot<Self::Tensor>>;
    /// Returns static observation/action space metadata.
    fn env_description(&self) -> EnvDescription<Self::Tensor>;
}
// ANCHOR_END: env

/// Tensor type used by an [`Env`] implementation.
pub type EnvTensor<E> = <E as Env>::Tensor;

/// Factory for constructing environments of one compatible type.
pub trait EnvBuilderTrait: Sync + Send + 'static {
    /// Tensor type used by environments produced by this builder.
    type Tensor: R2lTensor;
    /// Environment type produced by this builder.
    type Env: Env<Tensor = Self::Tensor>;

    /// Builds a fresh environment instance.
    fn build_env(&self) -> Result<Self::Env>;

    /// Returns the environment description for produced environments.
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

/// Collection of environment builders used to create rollout workers.
pub enum EnvBuilder<EB: EnvBuilderTrait> {
    /// Reuses one builder for `n_envs` homogeneous workers.
    Homogenous { builder: Arc<EB>, n_envs: usize },
    /// Uses one builder per worker.
    Heterogenous { builders: Vec<Arc<EB>> },
}

impl<EB: EnvBuilderTrait> EnvBuilder<EB> {
    /// Creates a homogeneous builder collection.
    pub fn homogenous(builder: EB, n_envs: usize) -> Self {
        Self::Homogenous {
            builder: Arc::new(builder),
            n_envs,
        }
    }

    /// Builds the environment at `idx`.
    pub fn build_idx(&self, idx: usize) -> Result<EB::Env> {
        match &self {
            Self::Homogenous { builder, .. } => builder.build_env(),
            Self::Heterogenous { builders } => builders[idx].build_env(),
        }
    }

    /// Returns the number of environments represented by this builder.
    pub fn num_envs(&self) -> usize {
        match self {
            Self::Homogenous { n_envs, .. } => *n_envs,
            Self::Heterogenous { builders } => builders.len(),
        }
    }

    /// Returns a representative environment builder.
    pub fn env_builder(&self) -> Arc<EB> {
        match &self {
            Self::Homogenous { builder, .. } => builder.clone(),
            Self::Heterogenous { builders } => builders[0].clone(),
        }
    }

    /// Returns a representative environment description.
    pub fn env_description(&self) -> Result<EnvDescription<EB::Tensor>> {
        match &self {
            Self::Homogenous { builder, n_envs: _ } => builder.env_description(),
            Self::Heterogenous { builders } => builders[0].env_description(),
        }
    }
}
