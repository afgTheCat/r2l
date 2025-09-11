// TODO: do we even need a folder for this?
pub mod categorical_distribution;
pub mod diagonal_distribution;

use anyhow::Result;
use bincode::{Decode, Encode};
use candle_core::Tensor;
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
    type Tensor = Tensor;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.get_action(observation),
            Self::DiagGaussian(diag) => diag.get_action(observation),
        }
    }

    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Tensor> {
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

impl Encode for DistributionKind {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        match self {
            Self::Categorical(distr) => {
                0u32.encode(encoder)?;
                distr.encode(encoder)
            }
            Self::DiagGaussian(distr) => {
                1u32.encode(encoder)?;
                distr.encode(encoder)
            }
        }
    }
}

impl Decode<()> for DistributionKind {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        let enum_type = u32::decode(decoder)?;
        match enum_type {
            0 => {
                let distr: CategoricalDistribution = CategoricalDistribution::decode(decoder)?;
                Ok(Self::Categorical(distr))
            }
            1 => {
                let distr: DiagGaussianDistribution = DiagGaussianDistribution::decode(decoder)?;
                Ok(Self::DiagGaussian(distr))
            }
            _ => unreachable!(),
        }
    }
}
