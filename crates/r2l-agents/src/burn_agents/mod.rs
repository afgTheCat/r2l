pub mod a2c;
pub mod ppo;

use burn::tensor::{Tensor as BurnTensor, backend::AutodiffBackend};

fn uplift_tensor<const N: usize, B: AutodiffBackend>(
    tensor: &BurnTensor<B::InnerBackend, N>,
) -> BurnTensor<B, N> {
    BurnTensor::from_data(tensor.to_data(), &Default::default())
}
