pub mod burn;
pub mod candle;

use crate::agents::ppo::{
    burn::PPOBurnLearningModuleBuilder, candle::PPOCandleLearningModuleBuilder,
};

// Unified builder trait that either builds a BurnPPO or a CandlePPO
// this is going to be massive xdddd
pub enum PPOBuiler {
    PPOBurnLearningModuleBuilder(PPOBurnLearningModuleBuilder),
    PPOCandleLearningModuleBuilder(PPOCandleLearningModuleBuilder),
}
