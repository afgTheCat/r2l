use burn::{Tensor as BurnTensor, module::Module, prelude::Backend};
use r2l_core::models::{Actor, Policy};

use crate::distributions::{
    categorical_distribution::CategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution,
};
pub mod categorical_distribution;
pub mod diagonal_distribution;

#[derive(Debug, Module)]
pub enum BurnPolicyKind<B: Backend> {
    Categorical(CategoricalDistribution<B>),
    Diag(DiagGaussianDistribution<B>),
}

impl<B: Backend> Actor for BurnPolicyKind<B> {
    type Tensor = BurnTensor<B, 1>;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.get_action(observation),
            Self::Diag(diag) => diag.get_action(observation),
        }
    }
}

impl<B: Backend> Policy for BurnPolicyKind<B> {
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

    fn entropy(&self, states: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::Diag(diag) => diag.entropy(states),
        }
    }

    fn resample_noise(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::Diag(diag) => diag.resample_noise(),
        }
    }
}
