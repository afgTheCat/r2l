use candle_core::{Device, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use itertools::izip;

use crate::Shape;
use crate::{
    error::{Error, TensorError},
    tensor::{R2lTensor, VecTensor},
};

type Result<T> = std::result::Result<T, TensorError>;

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        TensorError::operation("Candle backend", error).into()
    }
}

impl R2lTensor for Tensor {
    fn to_vec(&self) -> Result<Vec<f32>> {
        self.flatten_all()
            .and_then(|tensor| tensor.to_vec1())
            .map_err(|error| TensorError::operation("convert to vector", error))
    }

    fn to_shape(&self) -> Shape {
        self.shape().dims().into()
    }

    fn from_slice_and_shape(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        validate_shape(data.len(), &shape)?;
        Tensor::from_slice(data, shape.dims(), &Device::Cpu)
            .map_err(|error| TensorError::operation("construct from slice", error))
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        validate_shape(data.len(), &shape)?;
        Tensor::from_vec(data, shape.dims(), &Device::Cpu)
            .map_err(|error| TensorError::operation("construct from vector", error))
    }

    fn add(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "add")?;
        self.add(other)
            .map_err(|error| TensorError::operation("add", error))
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "subtract")?;
        self.sub(other)
            .map_err(|error| TensorError::operation("subtract", error))
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "multiply")?;
        self.mul(other)
            .map_err(|error| TensorError::operation("multiply", error))
    }

    fn gather(&self, dim: usize, indices: &Self) -> Result<Self> {
        super::validate_gather(self.dims(), indices.dims(), dim)?;
        indices
            .to_dtype(candle_core::DType::U32)
            .and_then(|indices| self.gather(&indices, dim))
            .map_err(|error| TensorError::operation("gather", error))
    }

    fn exp(&self) -> Result<Self> {
        self.exp()
            .map_err(|error| TensorError::operation("exponential", error))
    }

    fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        self.clamp(min, max)
            .map_err(|error| TensorError::operation("clamp", error))
    }

    fn minimum(&self, other: &Self) -> Result<Self> {
        ensure_same_shape(self, other, "minimum")?;
        Self::minimum(self, other).map_err(|error| TensorError::operation("minimum", error))
    }

    fn neg(&self) -> Result<Self> {
        self.neg()
            .map_err(|error| TensorError::operation("negate", error))
    }

    fn mean(&self) -> Result<Self> {
        if self.elem_count() == 0 {
            return Err(TensorError::EmptyInput {
                operation: "mean".into(),
            });
        }
        self.mean_all()
            .map_err(|error| TensorError::operation("mean", error))
    }

    fn sqr(&self) -> Result<Self> {
        self.sqr()
            .map_err(|error| TensorError::operation("square", error))
    }

    fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        Tensor::zeros(shape.dims(), candle_core::DType::F32, &Device::Cpu)
            .map_err(|error| TensorError::operation("create zeros", error))
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        let scalar = Tensor::full(scalar, (), self.device())
            .map_err(|error| TensorError::operation("create scalar", error))?;
        self.broadcast_mul(&scalar)
            .map_err(|error| TensorError::operation("multiply by scalar", error))
    }

    // TODO: we really need to check dim={0,1}, cuz I don't realy know what that means tbh
    fn softmax(&self, dim: usize) -> Result<Self> {
        softmax(self, dim).map_err(|err| TensorError::operation("softmax", err))
    }

    fn log_softmax(&self, dim: usize) -> super::Result<Self> {
        log_softmax(self, dim).map_err(|err| TensorError::operation("softmax", err))
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

fn ensure_same_shape(left: &Tensor, right: &Tensor, operation: &str) -> Result<()> {
    let left = left.shape().dims().to_vec();
    let right = right.shape().dims().to_vec();
    if left != right {
        return Err(TensorError::ShapeMismatch {
            operation: operation.into(),
            left: left.into(),
            right: right.into(),
        });
    }
    Ok(())
}

impl VecTensor {
    /// Clamps each element between the corresponding values in `min` and `max`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor shapes differ or the result cannot be constructed.
    pub fn clamp(&self, min: &Self, max: &Self) -> Result<Self> {
        self.ensure_same_shape(min, "clamp minimum")?;
        self.ensure_same_shape(max, "clamp maximum")?;
        let data = izip!(&self.data, &min.data, &max.data)
            .map(|(value, min, max)| value.clamp(*min, *max))
            .collect();
        Self::new(data, self.shape.clone())
    }
}
