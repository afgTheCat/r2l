use candle_core::{Device, Tensor};

use crate::tensor::{R2lTensor, R2lTensorMath, TensorData};

impl From<TensorData> for Tensor {
    fn from(val: TensorData) -> Self {
        let TensorData { data, shape } = val;
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }
}

impl From<Tensor> for TensorData {
    fn from(value: Tensor) -> Self {
        let shape = value.shape().clone().into_dims();
        let data: Vec<f32> = value.to_vec1().unwrap();
        Self { data, shape }
    }
}

impl R2lTensor for Tensor {
    fn to_vec(&self) -> Vec<f32> {
        self.to_vec1().unwrap()
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
        let t: Tensor = self.clone().into();
        let min_t = min.clone().into();
        let max_t = max.clone().into();
        (t.clamp(&min_t, &max_t).unwrap()).into()
    }
}
