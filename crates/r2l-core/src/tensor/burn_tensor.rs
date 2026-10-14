use burn::tensor::{
    Device, Tensor, TensorData as BurnTensorData,
    activation::{log_softmax, softmax},
};

use crate::Shape;
use crate::{
    error::TensorError,
    tensor::{DistributionTensor, OnPolicyTensor, R2lTensor, TensorOps},
};

type Result<T> = std::result::Result<T, TensorError>;

impl<const D: usize> R2lTensor for Tensor<D> {
    fn to_vec(&self) -> Result<Vec<f32>> {
        self.to_data()
            .try_to_vec()
            .map_err(|error| TensorError::operation("convert to vector", error))
    }

    fn to_shape(&self) -> Shape {
        self.shape().to_vec().into()
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate_len(data.len())?;
        if shape.rank() != D {
            return Err(TensorError::InvalidRank {
                expected: D,
                actual: shape.rank(),
            });
        }
        let data = BurnTensorData::new(data, shape.dims());
        Ok(Tensor::from_data(data, &Device::default()))
    }

    fn add(&self, other: &Self) -> Result<Self> {
        self.to_shape().ensure_equal(&other.to_shape(), "add")?;
        Ok(self.clone() + other.clone())
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        self.to_shape()
            .ensure_equal(&other.to_shape(), "subtract")?;
        Ok(self.clone() - other.clone())
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        self.to_shape()
            .ensure_equal(&other.to_shape(), "multiply")?;
        Ok(self.clone() * other.clone())
    }

    fn sqr(&self) -> Result<Self> {
        Ok(self.clone().powf_scalar(2.0))
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        Ok(self.clone().mul_scalar(scalar))
    }
}

impl<const D: usize> TensorOps for Tensor<D> {
    fn detach(&self) -> Self {
        self.clone().detach()
    }

    fn add_scalar(&self, scalar: f32) -> Result<Self> {
        Ok(self.clone().add_scalar(scalar))
    }

    fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        super::concatenated_shape(&tensors.iter().map(Self::to_shape).collect::<Vec<_>>(), dim)?;
        Ok(Tensor::cat(tensors.to_vec(), dim))
    }

    fn exp(&self) -> Result<Self> {
        Ok(self.clone().exp())
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
}

impl<const D: usize> OnPolicyTensor for Tensor<D> {
    fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        Ok(self.clone().clamp(min, max))
    }

    fn minimum(&self, other: &Self) -> Result<Self> {
        self.to_shape().ensure_equal(&other.to_shape(), "minimum")?;
        Ok(self.clone().min_pair(other.clone()))
    }
}

impl<const D: usize> DistributionTensor for Tensor<D> {
    fn matmul(&self, other: &Self) -> Result<Self> {
        super::matrix_dimensions(&self.to_shape(), &other.to_shape())?;
        Ok(self.clone().matmul(other.clone()))
    }

    fn log(&self) -> Result<Self> {
        Ok(self.clone().log())
    }

    fn from_vec_like(data: Vec<f32>, shape: impl Into<Shape>, like: &Self) -> Result<Self> {
        let shape = shape.into();
        shape.validate_len(data.len())?;
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

    fn sum_dim(&self, dim: usize) -> Result<Self> {
        super::validate_axis(&self.to_shape(), dim)?;
        Ok(self.clone().sum_dim(dim))
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        super::narrow_shape(&self.to_shape(), dim, start, length)?;
        Ok(self.clone().narrow(dim, start, length))
    }

    fn broadcast_as(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        super::validate_broadcast(&self.to_shape(), &shape)?;
        Ok(self.clone().expand(burn::tensor::Shape::from(shape.dims())))
    }

    fn log_sigmoid(&self) -> Result<Self> {
        Ok(burn::tensor::activation::log_sigmoid(self.clone()))
    }

    fn gather(&self, dim: usize, indices: &Self) -> Result<Self> {
        super::validate_gather(&self.to_shape(), &indices.to_shape(), dim)?;
        Ok(self.clone().gather(dim, indices.clone().int()))
    }

    fn softmax(&self, dim: usize) -> super::Result<Self> {
        Ok(softmax(self.clone(), dim))
    }

    fn log_softmax(&self, dim: usize) -> super::Result<Self> {
        Ok(log_softmax(self.clone(), dim))
    }
}
