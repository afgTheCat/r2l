// TODO: do we even need a folder for this?
pub mod categorical_distribution;
pub mod diagonal_distribution;

use std::{f32, fmt::Debug};

use anyhow::Result;
use candle_core::Tensor as CandleTensor;
use categorical_distribution::CategoricalDistribution;
use diagonal_distribution::DiagGaussianDistribution;
use r2l_core::models::{Actor, Policy};

#[derive(Debug, Clone)]
pub enum CandleDistributionKind {
    Categorical(CategoricalDistribution),
    DiagGaussian(DiagGaussianDistribution),
}

impl Actor for CandleDistributionKind {
    type Tensor = CandleTensor;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.get_action(observation),
            Self::DiagGaussian(diag) => diag.get_action(observation),
        }
    }
}

impl Policy for CandleDistributionKind {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(states, actions),
            Self::DiagGaussian(diag) => diag.log_probs(states, actions),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::DiagGaussian(diag) => diag.entropy(states),
        }
    }

    fn std(&self) -> Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::DiagGaussian(diag) => diag.std(),
        }
    }

    fn resample_noise(&mut self) -> Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::DiagGaussian(diag) => diag.resample_noise(),
        }
    }
}
