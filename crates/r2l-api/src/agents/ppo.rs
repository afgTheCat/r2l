pub mod burn;
pub mod candle;
pub mod unified;

use crate::{
    agents::ppo::{
        burn::{BurnBackend, BurnPPO},
        candle::CandlePPO,
    },
};
use ::burn::backend::NdArray;
use r2l_burn_lm::distributions::DistributionKind;
use r2l_candle_lm::distributions::CandleDistributionKind;
use r2l_core::{
    agents::Agent,
    distributions::Actor,
    sampler::buffer::{TrajectoryContainer, wrapper::BufferWrapper},
    tensor::R2lBuffer,
};
pub use unified::{PPOBackend, UnifiedPPOLearningModuleBuilder};

pub enum BurnOrCandlePPOActor {
    Burn(DistributionKind<NdArray>),
    Candle(CandleDistributionKind),
}

impl Actor for BurnOrCandlePPOActor {
    type Tensor = R2lBuffer;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Burn(d) => {
                let observation = observation.into();
                let action = d.get_action(observation)?;
                Ok(action.into())
            }
            Self::Candle(d) => {
                let observation = observation.into();
                let action = d.get_action(observation)?;
                Ok(action.into())
            }
        }
    }
}

pub enum BurnOrCandlePPO {
    Burn(BurnPPO<BurnBackend>),
    Candle(CandlePPO),
}

impl Agent for BurnOrCandlePPO {
    type Tensor = R2lBuffer;
    type Actor = BurnOrCandlePPOActor;

    fn actor(&self) -> Self::Actor {
        match self {
            Self::Burn(ppo) => {
                let actor = ppo.actor();
                BurnOrCandlePPOActor::Burn(actor)
            }
            Self::Candle(ppo) => {
                let actor = ppo.actor();
                BurnOrCandlePPOActor::Candle(actor)
            }
        }
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        match self {
            Self::Burn(ppo) => {
                let buffers = buffers
                    .as_ref()
                    .iter()
                    .map(|b| BufferWrapper::new(b))
                    .collect::<Vec<_>>();
                ppo.learn(&buffers)
            }
            Self::Candle(ppo) => {
                let buffers = buffers
                    .as_ref()
                    .iter()
                    .map(|b| BufferWrapper::new(b))
                    .collect::<Vec<_>>();
                ppo.learn(&buffers)
            }
        }
    }

    fn shutdown(&mut self) {
        match self {
            Self::Burn(ppo) => {
                ppo.shutdown();
            }
            Self::Candle(ppo) => {
                ppo.shutdown();
            }
        }
    }
}

pub type PPOLearningModuleBuilder = UnifiedPPOLearningModuleBuilder;
