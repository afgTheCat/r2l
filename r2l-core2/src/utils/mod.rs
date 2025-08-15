pub mod rollout_buffers;

use burn::{prelude::Backend, tensor::Tensor};

pub fn tensor_sqr<const N: usize, B: Backend>(t: Tensor<B, N>) -> Tensor<B, N> {
    t.clone() * t
}
