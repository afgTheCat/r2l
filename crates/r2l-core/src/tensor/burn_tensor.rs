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

    fn batch_mean(&self) -> anyhow::Result<Self> {
        let (data, shape) = self.to_vec_and_shape();
        anyhow::ensure!(
            !shape.is_empty(),
            "running mean update expects a batch dimension"
        );
        let batch_count = shape[0];
        let feature_shape = shape[1..].to_vec();
        let feature_size = feature_shape.iter().product::<usize>();
        anyhow::ensure!(
            batch_count * feature_size == data.len(),
            "batch tensor shape does not match data length"
        );

        let mut batch_mean = vec![0.0; feature_size];
        for sample in data.chunks_exact(feature_size) {
            for (mean, value) in batch_mean.iter_mut().zip(sample.iter()) {
                *mean += *value;
            }
        }
        for mean in &mut batch_mean {
            *mean /= batch_count as f32;
        }
        Ok(Self::from_vec_and_shape(batch_mean, feature_shape))
    }

    fn biased_var(&self) -> anyhow::Result<Self> {
        let (data, shape) = self.to_vec_and_shape();
        anyhow::ensure!(
            !shape.is_empty(),
            "running mean update expects a batch dimension"
        );
        let batch_count = shape[0];
        let feature_shape = shape[1..].to_vec();
        let feature_size = feature_shape.iter().product::<usize>();
        anyhow::ensure!(
            batch_count * feature_size == data.len(),
            "batch tensor shape does not match data length"
        );

        let batch_mean = self.batch_mean()?.to_vec();
        let mut batch_var = vec![0.0; feature_size];
        for sample in data.chunks_exact(feature_size) {
            for ((var, value), mean) in batch_var
                .iter_mut()
                .zip(sample.iter())
                .zip(batch_mean.iter())
            {
                let delta = *value - *mean;
                *var += delta * delta;
            }
        }
        for var in &mut batch_var {
            *var /= batch_count as f32;
        }
        Ok(Self::from_vec_and_shape(batch_var, feature_shape))
    }

    fn batch_count(&self) -> anyhow::Result<f32> {
        let (_, shape) = self.to_vec_and_shape();
        anyhow::ensure!(
            !shape.is_empty(),
            "running mean update expects a batch dimension"
        );
        Ok(shape[0] as f32)
    }

    fn mul_scalar(&self, scalar: f32) -> anyhow::Result<Self> {
        Ok(self.clone().mul_scalar(scalar))
    }
}
