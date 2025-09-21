use crate::tensor::{R2lBuffer, R2lTensor};
use burn::{
    prelude::Backend,
    tensor::{Tensor as BurnTensor, TensorData},
};

impl<B: Backend> From<R2lBuffer> for BurnTensor<B, 1> {
    fn from(value: R2lBuffer) -> Self {
        let device = Default::default();
        let tensor_data = TensorData::new(value.data, value.shape.clone());
        BurnTensor::from_data(tensor_data, &device)
    }
}

impl<B: Backend> From<BurnTensor<B, 1>> for R2lBuffer {
    fn from(value: BurnTensor<B, 1>) -> Self {
        let data = value.to_data().to_vec().unwrap();
        let shape = vec![data.len()];
        Self { data, shape }
    }
}

impl<B: Backend, const D: usize> R2lTensor for BurnTensor<B, D> {
    fn to_vec(&self) -> Vec<f32> {
        self.to_data().to_vec().unwrap()
    }
}
