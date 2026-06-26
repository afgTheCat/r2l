use candle_core::{Device, Tensor};

use crate::tensor::{R2lTensor, TensorData};

impl R2lTensor for Tensor {
    fn to_vec(&self) -> Vec<f32> {
        self.to_vec1().unwrap()
    }

    fn to_shape(&self) -> Vec<usize> {
        self.shape().dims().to_vec()
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

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

    fn zeros(shape: Vec<usize>) -> Self {
        Tensor::zeros(shape, candle_core::DType::F32, &Device::Cpu).unwrap()
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

#[cfg(test)]
mod test {
    use candle_core::Tensor;

    #[test]
    fn mean_things() {
        // what we have here is the following:
        let bm = Tensor::from_vec(
            // [0][0], [0][1], [1][0], [1][1]
            vec![1f32, 2., 3., 4.],
            vec![2, 2],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        dbg!(&bm);
        let m = bm.mean(0).unwrap();
        // result is 2, [0][0], [1][0]
        // result is 3, [0][1], [1][1]
        dbg!(&m);
        let m = bm.mean(1).unwrap();
        dbg!(&m);
    }
}
