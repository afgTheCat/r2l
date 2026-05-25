use candle_core::{Device, Tensor};

use crate::tensor::{R2lTensor, R2lTensorMath, RunningMeanTensor, TensorData};

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

impl RunningMeanTensor for Tensor {
    fn zeros(shape: Vec<usize>) -> Self {
        Tensor::zeros(shape, candle_core::DType::F32, &Device::Cpu).unwrap()
    }

    fn batch_mean(&self) -> anyhow::Result<Self> {
        Ok(self.mean(0)?)
    }

    fn biased_var(&self) -> anyhow::Result<Self> {
        let mean = self.mean_keepdim(0)?;
        let squares = self.broadcast_sub(&mean)?.sqr()?;
        Ok((squares.sum_keepdim(0)? / self.dim(0)? as f64)?.squeeze(0)?)
    }

    fn batch_count(&self) -> anyhow::Result<f32> {
        Ok(self.shape().dim(0)? as f32)
    }

    fn mul_scalar(&self, scalar: f32) -> anyhow::Result<Self> {
        let scalar = Tensor::full(scalar, (), self.device())?;
        Ok(self.broadcast_mul(&scalar)?)
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
