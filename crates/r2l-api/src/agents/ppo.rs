pub mod burn;
pub mod candle;

use ::burn::backend::{Autodiff, NdArray};
use r2l_core::agents::Agent;

use crate::agents::ppo::{
    burn::{BurnPPO, PPOBurnLearningModuleBuilder},
    candle::{CandlePPO, PPOCandleLearningModuleBuilder},
};

// Unified builder trait that either builds a BurnPPO or a CandlePPO
// this is going to be massive xdddd
pub enum PPOBuiler {
    PPOBurnLearningModuleBuilder(PPOBurnLearningModuleBuilder),
    PPOCandleLearningModuleBuilder(PPOCandleLearningModuleBuilder),
}
