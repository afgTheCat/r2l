pub mod distributions;
pub mod learning_module;
pub mod optimizer;
pub mod thread_safe_sequential;

use crate::learning_module::PolicyValuesLosses;
use r2l_core::policies::ModuleWithValueFunction;

pub trait CandleModuleWithValueFunction:
    ModuleWithValueFunction<
        InferenceTensor = candle_core::Tensor,
        Tensor = candle_core::Tensor,
        Losses = PolicyValuesLosses,
    >
{
}
