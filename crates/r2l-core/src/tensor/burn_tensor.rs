use burn::{
    prelude::Backend,
    tensor::{
        Tensor, TensorData as BurnTensorData,
        activation::{log_softmax, softmax},
    },
};

use crate::Shape;
use crate::{error::TensorError, tensor::R2lTensor};

type Result<T> = std::result::Result<T, TensorError>;

impl<const D: usize, B: Backend> R2lTensor for Tensor<B, D> {
    fn to_vec(&self) -> Result<Vec<f32>> {
        self.to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("convert to vector", error))
    }

    fn to_shape(&self) -> Shape {
        self.shape().to_vec().into()
    }

    fn from_slice_and_shape(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        validate_shape(data.len(), &shape)?;
        if shape.rank() != D {
            return Err(TensorError::InvalidRank {
                expected: D,
                actual: shape.rank(),
            });
        }
        let data = BurnTensorData::new(data.to_vec(), shape.dims());
        Ok(Tensor::from_data(data, &Default::default()))
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        validate_shape(data.len(), &shape)?;
        if shape.rank() != D {
            return Err(TensorError::InvalidRank {
                expected: D,
                actual: shape.rank(),
            });
        }
        let data = BurnTensorData::new(data, shape.dims());
        Ok(Tensor::from_data(data, &Default::default()))
    }

    fn from_vec_like(data: Vec<f32>, shape: impl Into<Shape>, like: &Self) -> Result<Self> {
        let shape = shape.into();
        validate_shape(data.len(), &shape)?;
        if shape.rank() != D {
            return Err(TensorError::InvalidRank {
                expected: D,
                actual: shape.rank(),
            });
        }
        Ok(Tensor::from_data(
            BurnTensorData::new(data, shape.dims()),
            &like.device(),
        ))
    }

    fn add_scalar(&self, scalar: f32) -> Result<Self> {
        Ok(self.clone().add_scalar(scalar))
    }

    fn sum_dim(&self, dim: usize) -> Result<Self> {
        super::validate_axis(&self.to_shape(), dim)?;
        Ok(self.clone().sum_dim(dim))
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        super::narrow_shape(&self.to_shape(), dim, start, length)?;
        Ok(self.clone().narrow(dim, start, length))
    }

    fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        super::concatenated_shape(&tensors.iter().map(Self::to_shape).collect::<Vec<_>>(), dim)?;
        Ok(Tensor::cat(tensors.to_vec(), dim))
    }

    fn broadcast_as(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        super::validate_broadcast(&self.to_shape(), &shape)?;
        Ok(self.clone().expand(burn::tensor::Shape::from(shape.dims())))
    }

    fn log_sigmoid(&self) -> Result<Self> {
        Ok(burn::tensor::activation::log_sigmoid(self.clone()))
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

    fn gather(&self, dim: usize, indices: &Self) -> Result<Self> {
        super::validate_gather(&self.to_shape(), &indices.to_shape(), dim)?;
        Ok(self.clone().gather(dim, indices.clone().int()))
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
        // Preserve the static rank while representing the all-element mean as one value.
        Ok(self.clone().mean().reshape([1; D]))
    }

    fn sqr(&self) -> Result<Self> {
        Ok(self.clone().powf_scalar(2.0))
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        Ok(self.clone().mul_scalar(scalar))
    }

    fn softmax(&self, dim: usize) -> super::Result<Self> {
        Ok(softmax(self.clone(), dim))
    }

    fn log_softmax(&self, dim: usize) -> super::Result<Self> {
        Ok(log_softmax(self.clone(), dim))
    }
}

fn validate_shape(data_len: usize, shape: &Shape) -> Result<()> {
    let expected = shape.num_elements();
    if expected != data_len {
        return Err(TensorError::InvalidShape {
            shape: shape.clone(),
            expected,
            actual: data_len,
        });
    }
    Ok(())
}

fn ensure_same_shape<const D: usize, B: Backend>(
    left: &Tensor<B, D>,
    right: &Tensor<B, D>,
    operation: &str,
) -> Result<()> {
    let left = left.shape().to_vec();
    let right = right.shape().to_vec();
    if left != right {
        return Err(TensorError::ShapeMismatch {
            operation: operation.into(),
            left: left.into(),
            right: right.into(),
        });
    }
    Ok(())
}
