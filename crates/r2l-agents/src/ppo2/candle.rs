use crate::ppo2::PPOModule2;
use candle_core::Tensor;
use r2l_candle_lm::learning_module::PolicyValuesLosses;

pub trait CandlePPOModule2:
    PPOModule2<InferenceTensor = Tensor, LearningTensor = Tensor>
{
}
