use candle_core::{Device, Tensor};
use candle_nn::ops::{log_softmax, softmax};

use crate::Shape;
use crate::{
    error::{Error, TensorError},
    tensor::{DistributionTensor, OnPolicyTensor, R2lTensor, TensorOps},
};

type Result<T> = std::result::Result<T, TensorError>;

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        TensorError::from(error).into()
    }
}

impl R2lTensor for Tensor {
    fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.flatten_all()?.to_vec1()?)
    }

    fn to_shape(&self) -> Shape {
        self.shape().dims().into()
    }

    fn from_slice_and_shape(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate_len(data.len())?;
        Ok(Tensor::from_slice(data, shape.dims(), &Device::Cpu)?)
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate_len(data.len())?;
        Ok(Tensor::from_vec(data, shape.dims(), &Device::Cpu)?)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        Ok(self.add(other)?)
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        Ok(self.sub(other)?)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        Ok(self.mul(other)?)
    }

    fn sqr(&self) -> Result<Self> {
        Ok(self.sqr()?)
    }

    fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        Tensor::zeros(shape.dims(), candle_core::DType::F32, &Device::Cpu)
            .expect("failed to create zero-filled Candle tensor")
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        let scalar = Tensor::full(scalar, (), self.device())?;
        Ok(self.broadcast_mul(&scalar)?)
    }
}

impl TensorOps for Tensor {
    fn detach(&self) -> Self {
        self.detach()
    }

    fn add_scalar(&self, scalar: f32) -> Result<Self> {
        Ok(self.affine(1., f64::from(scalar))?)
    }

    fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        super::concatenated_shape(&tensors.iter().map(Self::to_shape).collect::<Vec<_>>(), dim)?;
        Ok(Tensor::cat(tensors, dim)?)
    }

    fn exp(&self) -> Result<Self> {
        Ok(self.exp()?)
    }

    fn neg(&self) -> Result<Self> {
        Ok(self.neg()?)
    }

    fn mean(&self) -> Result<Self> {
        if self.elem_count() == 0 {
            return Err(TensorError::EmptyInput {
                operation: "mean".into(),
            });
        }
        Ok(self.mean_all()?)
    }
}

impl OnPolicyTensor for Tensor {
    fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        Ok(self.clamp(min, max)?)
    }

    fn minimum(&self, other: &Self) -> Result<Self> {
        Ok(Self::minimum(self, other)?)
    }
}

impl DistributionTensor for Tensor {
    fn matmul(&self, other: &Self) -> Result<Self> {
        super::matrix_dimensions(&self.to_shape(), &other.to_shape())?;
        Ok(self.matmul(other)?)
    }

    fn log(&self) -> Result<Self> {
        Ok(self.log()?)
    }

    fn from_vec_like(data: Vec<f32>, shape: impl Into<Shape>, like: &Self) -> Result<Self> {
        let shape = shape.into();
        shape.validate_len(data.len())?;
        Ok(Tensor::from_vec(data, shape.dims(), like.device())?)
    }

    fn sum_dim(&self, dim: usize) -> Result<Self> {
        Ok(self.sum_keepdim(dim)?)
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        super::narrow_shape(&self.to_shape(), dim, start, length)?;
        Ok(Tensor::narrow(self, dim, start, length)?)
    }

    fn broadcast_as(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        super::validate_broadcast(&self.to_shape(), &shape)?;
        Ok(Tensor::broadcast_as(self, shape.dims())?)
    }

    fn log_sigmoid(&self) -> Result<Self> {
        // log-softmax([0, x]) gives log(sigmoid(x)); its backward pass is stable at zero too.
        let zeros = self.zeros_like()?;
        let stacked = Tensor::stack(&[&zeros, self], self.rank())?;
        Ok(log_softmax(&stacked, self.rank())?
            .narrow(self.rank(), 1, 1)?
            .squeeze(self.rank())?)
    }

    fn gather(&self, dim: usize, indices: &Self) -> Result<Self> {
        super::validate_gather(self.dims(), indices.dims(), dim)?;
        Ok(self.gather(&indices.to_dtype(candle_core::DType::U32)?, dim)?)
    }

    // TODO: we really need to check dim={0,1}, cuz I don't realy know what that means tbh
    fn softmax(&self, dim: usize) -> Result<Self> {
        Ok(softmax(self, dim)?)
    }

    fn log_softmax(&self, dim: usize) -> super::Result<Self> {
        Ok(log_softmax(self, dim)?)
    }
}
