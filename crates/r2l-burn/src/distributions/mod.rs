use crate::distributions::{
    categorical_distribution::CategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution,
};
use burn::{Tensor as BurnTensor, module::Module, prelude::Backend};
use r2l_core::distributions::{Actor, Policy};
pub mod categorical_distribution;
pub mod diagonal_distribution;

#[derive(Debug, Module)]
pub enum DistributionKind<B: Backend> {
    Categorical(CategoricalDistribution<B>),
    Diag(DiagGaussianDistribution<B>),
}

impl<B: Backend> Actor for DistributionKind<B> {
    type Tensor = BurnTensor<B, 1>;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.get_action(observation),
            Self::Diag(diag) => diag.get_action(observation),
        }
    }
}

impl<B: Backend> Policy for DistributionKind<B> {
    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(observations, actions),
            Self::Diag(diag) => diag.log_probs(observations, actions),
        }
    }

    fn std(&self) -> anyhow::Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::Diag(diag) => diag.std(),
        }
    }

    fn entropy(&self) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(),
            Self::Diag(diag) => diag.entropy(),
        }
    }

    fn resample_noise(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::Diag(diag) => diag.resample_noise(),
        }
    }
}
