use candle_core::Tensor;
use r2l_candle_lm::learning_module2::PolicyValuesLosses;
use r2l_core::{
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
};

pub mod a2c;
pub mod ppo;
pub mod ppo2;
pub mod vpg;

pub trait ModuleWithValueFunction {
    type P: Policy<Tensor = Tensor>;
    type L: LearningModule<Losses = PolicyValuesLosses>;
    type V: ValueFunction<Tensor = Tensor>;

    fn get_inference_policy(&self) -> Self::P;

    fn get_policy_ref(&self) -> &Self::P;

    fn learning_module(&mut self) -> &mut Self::L;

    fn value_func(&self) -> &Self::V;
}
