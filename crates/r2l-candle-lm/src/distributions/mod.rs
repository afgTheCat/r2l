// TODO: do we even need a folder for this?
pub mod categorical_distribution;
pub mod diagonal_distribution;

use anyhow::Result;
use candle_core::Tensor as CandleTensor;
use categorical_distribution::CategoricalDistribution;
use diagonal_distribution::DiagGaussianDistribution;
use r2l_core::distributions::Policy;
use std::{f32, fmt::Debug};

#[derive(Debug, Clone)]
pub enum DistributionKind {
    Categorical(CategoricalDistribution),
    DiagGaussian(DiagGaussianDistribution),
}

impl Policy for DistributionKind {
    type Tensor = CandleTensor;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.get_action(observation),
            Self::DiagGaussian(diag) => diag.get_action(observation),
        }
    }

    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(states, actions),
            Self::DiagGaussian(diag) => diag.log_probs(states, actions),
        }
    }

    fn entropy(&self) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(),
            Self::DiagGaussian(diag) => diag.entropy(),
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
