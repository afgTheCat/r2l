use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData as BurnTensorData},
};

use crate::tensor::R2lTensor;

impl<B: Backend> R2lTensor for Tensor<B, 1> {
    fn to_vec(&self) -> Vec<f32> {
        self.to_data().to_vec().unwrap()
    }

    fn to_shape(&self) -> Vec<usize> {
        self.shape().into()
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let data = BurnTensorData::new(data, shape);
        Tensor::from_data(data, &Default::default())
    }

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

    fn zeros(shape: Vec<usize>) -> Self {
        let data = BurnTensorData::new(vec![0.0; shape.iter().product()], shape);
        Tensor::from_data(data, &Default::default())
    }

    fn mul_scalar(&self, scalar: f32) -> anyhow::Result<Self> {
        Ok(self.clone().mul_scalar(scalar))
    }
}
