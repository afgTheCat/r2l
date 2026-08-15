use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData as BurnTensorData},
};

use crate::{error::TensorError, tensor::R2lTensor};

type Result<T> = std::result::Result<T, TensorError>;

impl<B: Backend> R2lTensor for Tensor<B, 1> {
    fn to_vec(&self) -> Result<Vec<f32>> {
        self.to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("convert to vector", error))
    }

    fn to_shape(&self) -> Vec<usize> {
        self.shape().into()
    }

    fn from_slice_and_shape(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        validate_shape(data.len(), &shape)?;
        let data = BurnTensorData::new(data.to_vec(), shape);
        Ok(Tensor::from_data(data, &Default::default()))
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        validate_shape(data.len(), &shape)?;
        let data = BurnTensorData::new(data, shape);
        Ok(Tensor::from_data(data, &Default::default()))
    }

    fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "add")?;
        Ok(self.clone() + other.clone())
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "subtract")?;
        Ok(self.clone() - other.clone())
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "multiply")?;
        Ok(self.clone() * other.clone())
    }

    fn exp(&self) -> Result<Self> {
        Ok(self.clone().exp())
    }

    fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        Ok(self.clone().clamp(min, max))
    }

    fn minimum(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "minimum")?;
        Ok(self.clone().min_pair(other.clone()))
    }

    fn neg(&self) -> Result<Self> {
        Ok(self.clone().neg())
    }

    fn mean(&self) -> Result<Self> {
        if self.shape().num_elements() == 0 {
            return Err(TensorError::EmptyInput {
                operation: "mean".into(),
            });
        }
        Ok(self.clone().mean())
    }

    fn sqr(&self) -> Result<Self> {
        Ok(self.clone().powf_scalar(2.0))
    }

    fn zeros(shape: Vec<usize>) -> Result<Self> {
        let data = BurnTensorData::new(vec![0.0; shape.iter().product()], shape);
        Ok(Tensor::from_data(data, &Default::default()))
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        Ok(self.clone().mul_scalar(scalar))
    }
}

fn validate_shape(data_len: usize, shape: &[usize]) -> Result<()> {
    let expected = shape.iter().product();
    if expected != data_len {
        return Err(TensorError::InvalidShape {
            shape: shape.to_vec(),
            expected,
            actual: data_len,
        });
    }
    Ok(())
}

fn ensure_same_shape<B: Backend>(
    left: &Tensor<B, 1>,
    right: &Tensor<B, 1>,
    operation: &str,
) -> Result<()> {
    let left = left.shape().to_vec();
    let right = right.shape().to_vec();
    if left != right {
        return Err(TensorError::ShapeMismatch {
            operation: operation.into(),
            left,
            right,
        });
    }
    Ok(())
}
