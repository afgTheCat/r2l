// TODO: burn support is highly experimental, and we need to figure out the right abstractions at
// one point. Maybe next release.
pub mod burn_rollout_buffer;
pub mod distributions;
pub mod learning_module;
pub mod sequential;
pub mod tensors;

use burn::optim::AdamW;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::backend::AutodiffBackend;
use burn::{module::Module, prelude::Backend, tensor::Tensor};
use std::f32;
