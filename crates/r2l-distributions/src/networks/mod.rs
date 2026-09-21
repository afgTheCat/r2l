use std::fmt::Debug;

use r2l_core::tensor::R2lTensor;

mod cnn;
mod mlp;

pub trait Network: Send + Debug + Clone + 'static {
    type Tensor: R2lTensor;

    fn forward(&self, t: Self::Tensor) -> Self::Tensor;
}
