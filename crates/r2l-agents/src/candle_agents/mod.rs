pub mod a2c;
pub mod ppo;
pub mod ppo3;
pub mod ppo4;
pub mod vpg;

use candle_core::Tensor as CandleTensor;
use r2l_candle_lm::learning_module2::PolicyValuesLosses;
use r2l_core::{
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
};

pub trait ModuleWithValueFunction {
    type P: Policy<Tensor = CandleTensor>;
    type L: LearningModule<Losses = PolicyValuesLosses>;
    type V: ValueFunction<Tensor = CandleTensor>;

    fn get_inference_policy(&self) -> Self::P;

    fn get_policy_ref(&self) -> &Self::P;

    fn learning_module(&mut self) -> &mut Self::L;

    fn value_func(&self) -> &Self::V;
}
