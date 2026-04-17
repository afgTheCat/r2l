use burn::{
    prelude::Backend,
    tensor::{Tensor as BurnTensor, TensorData as BurnTensorData, backend::AutodiffBackend},
};

use crate::tensor::{R2lTensor, R2lTensorMath, TensorData};

impl<B: Backend> From<TensorData> for BurnTensor<B, 1> {
    fn from(value: TensorData) -> Self {
        let device = Default::default();
        let tensor_data = BurnTensorData::new(value.data, value.shape.clone());
        BurnTensor::from_data(tensor_data, &device)
    }
}

impl<B: Backend> From<BurnTensor<B, 1>> for TensorData {
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

impl<B: AutodiffBackend> R2lTensorMath for burn::Tensor<B, 1> {
    fn add(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(self.clone() + other.clone())
    }

    fn sub(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(self.clone() - other.clone())
    }

    fn mul(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(self.clone() * other.clone())
    }

    fn exp(&self) -> anyhow::Result<Self> {
        Ok(self.clone().exp())
    }

    fn clamp(&self, min: f32, max: f32) -> anyhow::Result<Self> {
        Ok(self.clone().clamp(min, max))
    }

    fn minimum(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(self.clone().min_pair(other.clone()))
    }

    fn neg(&self) -> anyhow::Result<Self> {
        Ok(self.clone().neg())
    }

    fn mean(&self) -> anyhow::Result<Self> {
        Ok(self.clone().mean())
    }

    fn sqr(&self) -> anyhow::Result<Self> {
        Ok(self.clone().powf_scalar(2.0))
    }
}
