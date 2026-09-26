use r2l_core::{
    Shape,
    env::Space,
    error::{Error, Result},
    models::Actor,
    tensor::R2lTensor,
};

use super::TensorParameter;
use super::{
    bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical,
};
use crate::{Network, Policy};

/// Policy variants that can be combined into a tuple or dictionary action.
#[derive(Debug, Clone)]
pub enum DistributionKind<N: Network, P: TensorParameter<N::Tensor> = <N as Network>::Tensor> {
    Categorical(Categorical<N>),
    DiagGaussian(DiagGaussian<N, P>),
    MultiBernoulli(MultiBernoulli<N>),
    MultiCategorical(MultiCategorical<N>),
    Composite(Composite<N, P>),
}

impl<N: Network, P: TensorParameter<N::Tensor>> DistributionKind<N, P> {
    /// Builds distributions for an action space using caller-supplied parameter factories.
    ///
    /// `network` builds one network for each leaf, receiving its parameter prefix
    /// and output width. `log_std` creates `[1, actions]` Gaussian parameters.
    /// Dictionary children follow the action space's ordered key iteration.
    ///
    /// # Errors
    /// Returns an error for invalid spaces or failed network/parameter construction.
    pub fn from_space<T: R2lTensor>(
        space: Space<T>,
        network: &mut impl FnMut(&str, usize) -> Result<N>,
        log_std: &mut impl FnMut(&str, usize) -> Result<P>,
    ) -> Result<Self> {
        Self::from_space_at(space, "policy", network, log_std)
    }

    fn from_space_at<T: R2lTensor>(
        space: Space<T>,
        prefix: &str,
        network: &mut impl FnMut(&str, usize) -> Result<N>,
        log_std: &mut impl FnMut(&str, usize) -> Result<P>,
    ) -> Result<Self> {
        Ok(match space {
            Space::Discrete(choices) => {
                Self::Categorical(Categorical::new(network(prefix, choices)?)?)
            }
            Space::Box { shape, .. } => {
                let width = shape.num_elements();
                Self::DiagGaussian(DiagGaussian::new(
                    network(prefix, width)?,
                    log_std(prefix, width)?,
                )?)
            }
            Space::MultiBinary { shape } => {
                Self::MultiBernoulli(MultiBernoulli::new(network(prefix, shape.num_elements())?)?)
            }
            Space::MultiDiscrete { nvec, .. } => {
                let counts = nvec.to_vec()?;
                if counts.is_empty()
                    || counts
                        .iter()
                        .any(|n| !n.is_finite() || *n < 1.0 || n.fract() != 0.0)
                {
                    return Err(Error::invalid_parameter(
                        "category counts",
                        "positive integers",
                        format!("{counts:?}"),
                    ));
                }
                let counts: Vec<_> = counts.into_iter().map(|n| n as usize).collect();
                Self::MultiCategorical(MultiCategorical::new(
                    network(prefix, counts.iter().sum())?,
                    counts,
                )?)
            }
            Space::Dict(spaces) => Self::from_space_at(
                Space::Tuple(spaces.into_values().collect()),
                prefix,
                network,
                log_std,
            )?,
            Space::Tuple(spaces) => Self::Composite(Composite::new(
                spaces
                    .into_iter()
                    .enumerate()
                    .map(|(index, space)| {
                        Self::from_space_at(space, &format!("{prefix}.{index}"), network, log_std)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )?),
        })
    }

    fn inner(&self) -> &dyn Policy<Tensor = N::Tensor> {
        match self {
            Self::Categorical(policy) => policy,
            Self::DiagGaussian(policy) => policy,
            Self::MultiBernoulli(policy) => policy,
            Self::MultiCategorical(policy) => policy,
            Self::Composite(policy) => policy,
        }
    }
}

impl<N: Network, P: TensorParameter<N::Tensor>> Actor for DistributionKind<N, P> {
    type Tensor = N::Tensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        self.inner().action(observation)
    }
    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        self.inner().mode_action(observation)
    }
}

impl<N: Network, P: TensorParameter<N::Tensor>> Policy for DistributionKind<N, P> {
    fn action_shape(&self) -> Shape {
        self.inner().action_shape()
    }
    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        self.inner().log_probs(observations, actions)
    }
    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        self.inner().entropy(observations)
    }
    fn std(&self) -> Result<Option<f32>> {
        self.inner().std()
    }
}
