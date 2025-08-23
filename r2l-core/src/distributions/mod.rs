// TODO: should distributions be their own crate?

pub mod categorical_distribution;
pub mod diagonal_distribution;

use bincode::{Decode, Encode};
use candle_core::{Result, Tensor};
use categorical_distribution::CategoricalDistribution;
use diagonal_distribution::DiagGaussianDistribution;
use std::{f32, fmt::Debug};

// TODO: Decoding will need a context to store what device we want the tensors to be decoded to
// The phylosophy behind this should be that distributions are stateless, therfore cloenable and
// 'static and self contained. Will see if we can stick to this, but it is the agent that has the
// liberty to not be stateless and such
pub trait Distribution: Sync + Debug + 'static {
    type Observation;
    type Action;
    type Entropy;

    fn get_action(&self, observation: Self::Observation) -> Result<Self::Action>;
    fn log_probs(&self, states: Self::Observation, actions: Self::Action) -> Result<Tensor>;
    fn std(&self) -> Result<f32>;
    fn entropy(&self) -> Result<Self::Entropy>;
    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub enum DistributionKind {
    Categorical(CategoricalDistribution),
    DiagGaussian(DiagGaussianDistribution),
}

impl Distribution for DistributionKind {
    type Observation = Tensor;
    type Action = Tensor;
    type Entropy = Tensor;

    fn get_action(&self, observation: Self::Observation) -> Result<Self::Action> {
        match self {
            Self::Categorical(cat) => cat.get_action(observation),
            Self::DiagGaussian(diag) => diag.get_action(observation),
        }
    }

    fn log_probs(&self, states: Self::Observation, actions: Self::Action) -> Result<Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(states, actions),
            Self::DiagGaussian(diag) => diag.log_probs(states, actions),
        }
    }

    fn entropy(&self) -> Result<Self::Entropy> {
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
