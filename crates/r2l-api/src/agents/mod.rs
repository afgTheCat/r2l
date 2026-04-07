pub mod ppo;

use crate::{
    agents::ppo::{
        burn::{BurnBackend, BurnPPO, PPOBurnLearningModuleBuilder},
        candle::{CandlePPO, PPOCandleLearningModuleBuilder},
    },
    builders::distribution::ActionSpaceType,
};
use burn::backend::NdArray;
use r2l_burn_lm::distributions::DistributionKind;
use r2l_candle_lm::distributions::CandleDistributionKind;
use r2l_core::{
    agents::Agent,
    distributions::Actor,
    sampler::buffer::{TrajectoryContainer, wrapper::BufferWrapper},
    tensor::R2lBuffer,
};

pub trait AgentBuilder {
    type Agent: Agent;

    // TODO: the arguments to this funciton may not be final
    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent>;
}

pub enum LearningModuleBuilder {
    Burn(PPOBurnLearningModuleBuilder),
    Candle(PPOCandleLearningModuleBuilder),
}

pub enum BurnOrCandlePPO {
    Burn(BurnPPO<BurnBackend>),
    Candle(CandlePPO),
}

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
