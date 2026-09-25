use crate::Shape;

pub mod normalizer;

use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::tensor::R2lTensor;

/// Description of an observation or action space.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        shape: Shape,
    },
    /// Multiple discrete spaces packed into one tensor.
    MultiDiscrete {
        /// Number of categories for each discrete dimension.
        nvec: T,
        /// Tensor shape of the discrete dimensions.
        shape: Shape,
    },
    /// Binary tensor space.
    MultiBinary {
        /// Tensor shape of the binary dimensions.
        shape: Shape,
    },
    /// Ordered collection of spaces.
    Tuple(Vec<Space<T>>),
    /// Named collection of spaces.
    Dict(BTreeMap<String, Space<T>>),
}

impl<T: R2lTensor> Space<T> {
    /// Converts tensor values in this space to another tensor type.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor cannot be converted.
    pub fn convert<U: R2lTensor>(&self) -> Result<Space<U>, Error> {
        Ok(match self {
            Self::Discrete(size) => Space::Discrete(*size),
            Self::Box { min, max, shape } => Space::Box {
                min: min.as_ref().map(U::convert).transpose()?,
                max: max.as_ref().map(U::convert).transpose()?,
                shape: shape.clone(),
            },
            Self::MultiDiscrete { nvec, shape } => Space::MultiDiscrete {
                nvec: U::convert(nvec)?,
                shape: shape.clone(),
            },
            Self::MultiBinary { shape } => Space::MultiBinary {
                shape: shape.clone(),
            },
            Self::Tuple(spaces) => {
                Space::Tuple(spaces.iter().map(Self::convert).collect::<Result<_, _>>()?)
            }
            Self::Dict(spaces) => Space::Dict(
                spaces
                    .iter()
                    .map(|(key, space)| Ok((key.clone(), space.convert()?)))
                    .collect::<Result<_, Error>>()?,
            ),
        })
    }

    /// Returns the Gymnasium shape when the space has one.
    pub fn shape(&self) -> Option<Shape> {
        match self {
            Self::Discrete(_) => Some(Shape::default()),
            Self::Box { shape, .. }
            | Self::MultiDiscrete { shape, .. }
            | Self::MultiBinary { shape } => Some(shape.clone()),
            Self::Tuple(_) | Self::Dict(_) => None,
        }
    }

    /// Returns the encoded shape of one observation, excluding the batch dimension.
    ///
    /// Discrete observations are one-hot vectors. Tuple and dictionary observations
    /// concatenate their encoded fields into a flat vector. Other spaces preserve
    /// their declared dimensions and axis order, including an empty shape for scalars.
    /// This describes the logical shape even when observations are stored flat;
    /// it does not imply an image channel layout or transpose the observation.
    #[must_use]
    pub fn observation_shape(&self) -> Shape {
        match self {
            Self::Discrete(_) | Self::Tuple(_) | Self::Dict(_) => Shape::from([self.size()]),
            Self::Box { shape, .. }
            | Self::MultiDiscrete { shape, .. }
            | Self::MultiBinary { shape } => shape.clone(),
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
            | Self::MultiBinary { shape, .. } => shape.num_elements(),
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
            | Self::MultiBinary { shape } => shape.num_elements(),
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

    /// Returns the encoded shape of one observation, excluding the batch dimension.
    ///
    /// See [`Space::observation_shape`] for the encoding of each space variant.
    #[must_use]
    pub fn observation_shape(&self) -> Shape {
        self.observation_space.observation_shape()
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
    fn reset(&mut self, seed: u64) -> Result<Self::Tensor, Error>;

    /// Applies one action and returns the resulting transition snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment cannot apply the action.
    fn step(&mut self, action: Self::Tensor) -> Result<Snapshot<Self::Tensor>, Error>;

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
    fn build_env(&self) -> Result<Self::Env, Error>;

    /// Returns the environment description for produced environments.
    ///
    /// # Errors
    ///
    /// Returns an error if a representative environment cannot be constructed.
    fn env_description(&self) -> Result<EnvDescription<<Self::Env as Env>::Tensor>, Error> {
        let env = self.build_env()?;
        Ok(env.env_description())
    }
}
// ANCHOR_END: env_builder

impl<E: Env, F: Sync + Send + 'static> EnvBuilder for F
where
    F: Fn() -> Result<E, Error>,
{
    type Env = E;

    fn build_env(&self) -> Result<E, Error> {
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
                return Err(Error::invalid_parameter(
                    "n_envs",
                    "a value greater than zero",
                    "0",
                ));
            }
            EnvBuilderKind::Heterogeneous { builders } if builders.is_empty() => {
                return Err(Error::invalid_parameter(
                    "builders",
                    "at least one environment builder",
                    "empty",
                ));
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
            return Err(Error::invalid_parameter(
                "environment index",
                format!("an index below {n_envs}"),
                idx.to_string(),
            ));
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
