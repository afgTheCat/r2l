use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use anyhow::Result;

use crate::tensor::R2lTensor;

/// Description of an observation or action space.
#[derive(Debug, Clone)]
pub enum Space<T: R2lTensor> {
    /// Discrete space with `usize` possible values.
    Discrete(usize),
    /// Gymnasium Box space with optional elementwise bounds.
    Box {
        /// Optional minimum values.
        min: Option<T>,
        /// Optional maximum values.
        max: Option<T>,
        /// Tensor shape of the space.
        shape: Vec<usize>,
    },
    /// Multiple discrete spaces packed into one tensor.
    MultiDiscrete {
        /// Number of categories for each discrete dimension.
        nvec: T,
        /// Tensor shape of the discrete dimensions.
        shape: Vec<usize>,
    },
    /// Binary tensor space.
    MultiBinary {
        /// Tensor shape of the binary dimensions.
        shape: Vec<usize>,
    },
    /// Ordered collection of spaces.
    Tuple(Vec<Space<T>>),
    /// Named collection of spaces.
    Dict(BTreeMap<String, Space<T>>),
}

impl<T: R2lTensor> Space<T> {
    /// Returns the Gymnasium shape when the space has one.
    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            Self::Discrete(_) => Some(&[]),
            Self::Box { shape, .. }
            | Self::MultiDiscrete { shape, .. }
            | Self::MultiBinary { shape } => Some(shape),
            Self::Tuple(_) | Self::Dict(_) => None,
        }
    }

    /// Returns the flattened size of the space.
    pub fn size(&self) -> usize {
        match &self {
            Self::Discrete(size) => *size,
            Self::Box { shape, .. }
            | Self::MultiDiscrete { shape, .. }
            | Self::MultiBinary { shape, .. } => shape.iter().product(),
            Self::Tuple(spaces) => spaces.iter().map(Self::size).sum(),
            Self::Dict(spaces) => spaces.values().map(Self::size).sum(),
        }
    }
}

/// Observation and action space metadata for an environment.
#[derive(Debug, Clone)]
pub struct EnvDescription<T: R2lTensor> {
    /// Space returned by [`Env::reset`] and [`Env::step`].
    pub observation_space: Space<T>,
    /// Space accepted by [`Env::step`].
    pub action_space: Space<T>,
}

impl<T: R2lTensor> EnvDescription<T> {
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
pub struct Snapshot<T: R2lTensor> {
    /// Observation after the action was applied.
    pub state: T,
    /// Reward produced by the transition.
    pub reward: f32,
    /// Whether the environment reached a terminal state.
    pub terminated: bool,
    /// Whether the episode ended because of a time limit or external cutoff.
    pub truncated: bool,
}

impl<T: R2lTensor> Snapshot<T> {
    pub fn new(state: T, reward: f32, terminated: bool, truncated: bool) -> Self {
        Self {
            state,
            reward,
            terminated,
            truncated,
        }
    }

    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Tensor type used by an [`Env`] implementation.
pub type EnvTensor<E> = <E as Env>::Tensor;

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

// ANCHOR: env_builder
/// Factory for constructing environments of one compatible type.
pub trait EnvBuilder: Sync + Send + 'static {
    /// Environment type produced by this builder.
    type Env: Env;

    /// Builds a fresh environment instance.
    fn build_env(&self) -> Result<Self::Env>;

    /// Returns the environment description for produced environments.
    fn env_description(&self) -> Result<EnvDescription<<Self::Env as Env>::Tensor>> {
        let env = self.build_env()?;
        Ok(env.env_description())
    }
}
// ANCHOR_END: env_builder

pub type TensorOfEnvBuilder<EB> = <<EB as EnvBuilder>::Env as Env>::Tensor;

impl<E: Env, F: Sync + Send + 'static> EnvBuilder for F
where
    F: Fn() -> Result<E>,
{
    type Env = E;

    fn build_env(&self) -> Result<Self::Env> {
        (self)()
    }
}

/// Collection of environment builders used to create rollout workers.
pub enum EnvBuilderType<EB: EnvBuilder> {
    /// Reuses one builder for `n_envs` homogeneous workers.
    Homogenous { builder: Arc<EB>, n_envs: usize },
    /// Uses one builder per worker.
    Heterogenous { builders: Vec<Arc<EB>> },
}

impl<EB: EnvBuilder> Clone for EnvBuilderType<EB> {
    fn clone(&self) -> Self {
        match self {
            Self::Homogenous { builder, n_envs } => Self::Homogenous {
                builder: builder.clone(),
                n_envs: *n_envs,
            },
            Self::Heterogenous { builders } => Self::Heterogenous {
                builders: builders.clone(),
            },
        }
    }
}

impl<EB: EnvBuilder> EnvBuilderType<EB> {
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
    pub fn env_description(&self) -> Result<EnvDescription<<EB::Env as Env>::Tensor>> {
        match &self {
            Self::Homogenous { builder, n_envs: _ } => builder.env_description(),
            Self::Heterogenous { builders } => builders[0].env_description(),
        }
    }
}

/// Returns `(offset, choices)` ranges for a flattened multi-discrete logits vector.
pub fn action_ranges(nvec: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    nvec.iter().scan(0, |offset, choices| {
        let start = *offset;
        *offset += *choices;
        Some((start, *choices))
    })
}
