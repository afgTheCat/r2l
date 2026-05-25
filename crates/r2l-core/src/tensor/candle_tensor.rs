use candle_core::{Device, Tensor};

use crate::tensor::{R2lTensor, R2lTensorMath, TensorData};

impl R2lTensor for Tensor {
    fn to_vec(&self) -> Vec<f32> {
        self.to_vec1().unwrap()
    }

    fn to_vec_and_shape(&self) -> (Vec<f32>, Vec<usize>) {
        (self.to_vec1().unwrap(), self.shape().dims().to_vec())
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }
}

impl R2lTensorMath for Tensor {
    fn add(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(self.add(other)?)
    }

    fn sub(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(self.sub(other)?)
    }

    fn mul(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(self.mul(other)?)
    }

    fn exp(&self) -> anyhow::Result<Self> {
        Ok(self.exp()?)
    }

    fn clamp(&self, min: f32, max: f32) -> anyhow::Result<Self> {
        Ok(self.clamp(min, max)?)
    }

    fn minimum(&self, other: &Self) -> anyhow::Result<Self> {
        Ok(Self::minimum(self, other)?)
    }

    fn neg(&self) -> anyhow::Result<Self> {
        Ok(self.neg()?)
    }

    fn mean(&self) -> anyhow::Result<Self> {
        Ok(self.mean_all()?)
    }

    fn sqr(&self) -> anyhow::Result<Self> {
        Ok(self.sqr()?)
    }
}

impl TensorData {
    // TODO: implement this without relying on candle
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let t = Tensor::convert(self);
        let min_t = Tensor::convert(min);
        let max_t = Tensor::convert(max);
        TensorData::convert(&t.clamp(&min_t, &max_t).unwrap())
    }
}
