use r2l_core::{Shape, error::Result, models::Actor};

use super::TensorParameter;
use super::{
    bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical,
};
use crate::{Network, Policy2};

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
    fn inner(&self) -> &dyn Policy2<Tensor = N::Tensor> {
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

impl<N: Network, P: TensorParameter<N::Tensor>> Policy2 for DistributionKind<N, P> {
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
