use r2l_core::policies::ModuleWithValueFunction;

use crate::learning_module::PolicyValuesLosses;

// TODO: burn support is highly experimental, and we need to figure out the right abstractions at
// one point. Maybe next release.
pub mod distributions;
pub mod learning_module;
pub mod sequential;

pub trait BurnModuleWithValueFunction<B: burn::tensor::backend::AutodiffBackend>:
    ModuleWithValueFunction<
        InferenceTensor = burn::tensor::Tensor<B::InnerBackend, 1>,
        Tensor = burn::tensor::Tensor<B, 1>,
        Losses = PolicyValuesLosses<B>,
    >
{
}
