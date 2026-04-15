use candle_core::{Device, Tensor as CandleTensor};

use crate::tensor::{R2lBuffer, R2lTensor, R2lTensorMath};

impl From<R2lBuffer> for CandleTensor {
    fn from(val: R2lBuffer) -> Self {
        let R2lBuffer { data, shape } = val;
        CandleTensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }
}

impl From<CandleTensor> for R2lBuffer {
    fn from(value: CandleTensor) -> Self {
        let shape = value.shape().clone().into_dims();
        let data: Vec<f32> = value.to_vec1().unwrap();
        Self { data, shape }
    }
}

impl R2lTensor for CandleTensor {
    fn to_vec(&self) -> Vec<f32> {
        self.to_vec1().unwrap()
    }
}

impl R2lTensorMath for CandleTensor {
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

impl R2lBuffer {
    // TODO: implement this without relying on candle
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let t: CandleTensor = self.clone().into();
        let min_t = min.clone().into();
        let max_t = max.clone().into();
        (t.clamp(&min_t, &max_t).unwrap()).into()
    }
}
