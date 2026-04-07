pub mod burn;
pub mod candle;

use r2l_core::{agents::Agent, distributions::Actor, tensor::R2lBuffer};

use crate::agents::ppo::{
    burn::{BurnBackend, BurnPPO, PPOBurnLearningModuleBuilder},
    candle::{CandlePPO, PPOCandleLearningModuleBuilder},
};
use ::burn::tensor::Tensor as BurnTensor;

// Unified builder trait that either builds a BurnPPO or a CandlePPO
// this is going to be massive xdddd
pub enum PPOBuiler {
    PPOBurnLearningModuleBuilder(PPOBurnLearningModuleBuilder),
    PPOCandleLearningModuleBuilder(PPOCandleLearningModuleBuilder),
}

pub enum PPO {
    Burn(BurnPPO<BurnBackend>),
    Candle(CandlePPO),
}

impl Actor for PPO {
    type Tensor = R2lBuffer;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Burn(ppo) => {
                let observation: BurnTensor<BurnBackend, 1> = observation.into();
                // let action = ppo
                todo!()
            }
            Self::Candle(ppo) => {
                todo!()
            }
        }
    }
}
