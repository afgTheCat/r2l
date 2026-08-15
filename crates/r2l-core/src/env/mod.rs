pub mod normalizer;

use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use crate::error::{Error, InvalidParameterError};
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

    /// Returns the flattened model width for this space.
    ///
    /// Discrete spaces use one-hot observations and categorical logits, so
    /// this returns their number of categories. Use [`Self::action_size`] for
    /// the width of an encoded action.
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

    /// Returns the number of scalar values used to encode an action.
    ///
    /// A discrete action is encoded as one category index, while discrete
    /// observations continue to use the one-hot width returned by [`Self::size`].
    #[must_use]
    pub fn action_size(&self) -> usize {
        match self {
            Self::Discrete(_) => 1,
            Self::Box { shape, .. }
            | Self::MultiDiscrete { shape, .. }
            | Self::MultiBinary { shape } => shape.iter().product(),
            Self::Tuple(spaces) => spaces.iter().map(Self::action_size).sum(),
            Self::Dict(spaces) => spaces.values().map(Self::action_size).sum(),
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
        self.action_space.action_size()
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
    /// Creates a transition snapshot.
    pub fn new(state: T, reward: f32, terminated: bool, truncated: bool) -> Self {
        Self {
            state,
            reward,
            terminated,
            truncated,
        }
    }

    /// Returns `true` when the transition ends the episode for any reason.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the environment cannot be reset.
    fn reset(&mut self, seed: u64) -> Result<Self::Tensor, crate::error::Error>;

    /// Applies one action and returns the resulting transition snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment cannot apply the action.
    fn step(&mut self, action: Self::Tensor)
    -> Result<Snapshot<Self::Tensor>, crate::error::Error>;

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
    ///
    /// # Errors
    ///
    /// Returns an error if the environment cannot be constructed.
    fn build_env(&self) -> Result<Self::Env, crate::error::Error>;

    /// Returns the environment description for produced environments.
    ///
    /// # Errors
    ///
    /// Returns an error if a representative environment cannot be constructed.
    fn env_description(
        &self,
    ) -> Result<EnvDescription<<Self::Env as Env>::Tensor>, crate::error::Error> {
        let env = self.build_env()?;
        Ok(env.env_description())
    }
}
// ANCHOR_END: env_builder

/// Tensor type produced by environments built by `EB`.
pub type TensorOfEnvBuilder<EB> = <<EB as EnvBuilder>::Env as Env>::Tensor;

impl<E: Env, F: Sync + Send + 'static> EnvBuilder for F
where
    F: Fn() -> Result<E, crate::error::Error>,
{
    type Env = E;

    fn build_env(&self) -> Result<E, crate::error::Error> {
        (self)()
    }
}

/// Validated, non-empty collection of environment builders used to create rollout workers.
pub struct EnvBuilderType<EB: EnvBuilder>(EnvBuilderKind<EB>);

enum EnvBuilderKind<EB: EnvBuilder> {
    /// Reuses one builder for `n_envs` homogeneous workers.
    Homogeneous {
        /// Shared environment builder.
        builder: Arc<EB>,
        /// Number of environments to construct.
        n_envs: usize,
    },
    /// Uses one builder per worker.
    Heterogeneous {
        /// Builders in worker-index order.
        builders: Vec<Arc<EB>>,
    },
}

impl<EB: EnvBuilder> Clone for EnvBuilderType<EB> {
    fn clone(&self) -> Self {
        Self(match &self.0 {
            EnvBuilderKind::Homogeneous { builder, n_envs } => EnvBuilderKind::Homogeneous {
                builder: builder.clone(),
                n_envs: *n_envs,
            },
            EnvBuilderKind::Heterogeneous { builders } => EnvBuilderKind::Heterogeneous {
                builders: builders.clone(),
            },
        })
    }
}

impl<EB: EnvBuilder> EnvBuilderType<EB> {
    fn from_kind(kind: EnvBuilderKind<EB>) -> Result<Self, Error> {
        match &kind {
            EnvBuilderKind::Homogeneous { n_envs: 0, .. } => {
                return Err(Error::InvalidParameter(Box::new(
                    InvalidParameterError::InvalidValue {
                        name: "n_envs".into(),
                        expected: "a value greater than zero".into(),
                        value: "0".into(),
                    },
                )));
            }
            EnvBuilderKind::Heterogeneous { builders } if builders.is_empty() => {
                return Err(Error::InvalidParameter(Box::new(
                    InvalidParameterError::InvalidValue {
                        name: "builders".into(),
                        expected: "at least one environment builder".into(),
                        value: "empty".into(),
                    },
                )));
            }
            _ => {}
        }
        Ok(Self(kind))
    }

    /// Creates a homogeneous builder collection.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_envs` is zero.
    pub fn homogeneous(builder: EB, n_envs: usize) -> Result<Self, Error> {
        Self::from_kind(EnvBuilderKind::Homogeneous {
            builder: Arc::new(builder),
            n_envs,
        })
    }

    /// Creates a heterogeneous collection with one environment per builder.
    ///
    /// # Errors
    ///
    /// Returns an error if `builders` is empty.
    pub fn heterogeneous(builders: Vec<EB>) -> Result<Self, Error> {
        Self::from_kind(EnvBuilderKind::Heterogeneous {
            builders: builders.into_iter().map(Arc::new).collect(),
        })
    }

    /// Builds the environment at `idx`.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected builder cannot construct an environment.
    pub fn build_idx(&self, idx: usize) -> Result<EB::Env, Error> {
        let n_envs = self.num_envs();
        if idx >= self.num_envs() {
            return Err(Error::InvalidParameter(Box::new(
                InvalidParameterError::InvalidValue {
                    name: "environment index".into(),
                    expected: format!("an index below {n_envs}"),
                    value: idx.to_string(),
                },
            )));
        }
        match &self.0 {
            EnvBuilderKind::Homogeneous { builder, .. } => builder.build_env(),
            EnvBuilderKind::Heterogeneous { builders } => builders[idx].build_env(),
        }
    }

    /// Returns the number of environments represented by this builder.
    #[must_use]
    pub fn num_envs(&self) -> usize {
        match &self.0 {
            EnvBuilderKind::Homogeneous { n_envs, .. } => *n_envs,
            EnvBuilderKind::Heterogeneous { builders } => builders.len(),
        }
    }

    /// Returns a representative environment description.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected builder cannot provide a description.
    pub fn env_description(&self) -> Result<EnvDescription<<EB::Env as Env>::Tensor>, Error> {
        match &self.0 {
            EnvBuilderKind::Homogeneous { builder, n_envs: _ } => builder.env_description(),
            EnvBuilderKind::Heterogeneous { builders } => builders[0].env_description(),
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
