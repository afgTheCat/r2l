// TODO: should distributions be their own crate?

pub mod categorical_distribution;
pub mod diagonal_distribution;

use bincode::{Decode, Encode};
use candle_core::{Result, Tensor};
use categorical_distribution::CategoricalDistribution;
use diagonal_distribution::DiagGaussianDistribution;
use enum_dispatch::enum_dispatch;
use std::{f32, fmt::Debug};

// TODO: Decoding will need a context to store what device we want the tensors to be decoded to
// The phylosophy behind this should be that distributions are stateless, therfore cloenable and
// 'static and self contained. Will see if we can stick to this, but it is the agent that has the
// liberty to not be stateless and such
#[enum_dispatch]
pub trait Distribution: Sync + Debug + 'static {
    fn get_action(&self, observation: &Tensor) -> Result<(Tensor, Tensor)>;
    fn log_probs(&self, states: &Tensor, actions: &Tensor) -> Result<Tensor>;
    fn std(&self) -> Result<f32>;
    fn entropy(&self) -> Result<Tensor>;
    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
#[enum_dispatch(Distribution)]
pub enum DistributionKind {
    Categorical(CategoricalDistribution),
    DiagGaussian(DiagGaussianDistribution),
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
