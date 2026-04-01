use r2l_core::policies::ModuleWithValueFunction;

use crate::learning_module::PolicyValuesLosses;

pub mod distributions;
pub mod learning_module;
pub mod optimizer;
pub mod thread_safe_sequential;

pub trait CandleModuleWithValueFunction:
    ModuleWithValueFunction<
        InferenceTensor = candle_core::Tensor,
        Tensor = candle_core::Tensor,
        Losses = PolicyValuesLosses,
    >
{
}
