pub mod burn;
pub mod candle;

use r2l_core::{agents::Agent, tensor::R2lBuffer};

use crate::agents::ppo::{
    burn::{BurnBackend, BurnPPO, PPOBurnLearningModuleBuilder},
    candle::{CandlePPO, PPOCandleLearningModuleBuilder},
};

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

// impl Agent for PPO {
//     type Tensor = R2lBuffer;
// }
