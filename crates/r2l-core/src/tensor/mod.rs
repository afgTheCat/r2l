#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

use crate::error::TensorError;

type Result<T> = std::result::Result<T, TensorError>;

// NOTE: we might want to add int_vec_and_shape method for less allocations, since to_vec_and_shape
// usually clones the inner vector. Would be useful in ActorWrapper.
//
/// Tensor contract shared by environments, policies, buffers, agents, and
/// built-in algorithm utilities.
///
/// Implementors should be cheap enough to clone for rollout storage and safe to
/// move across worker threads. `to_vec` is mainly for inspection, logging, and
/// simple environment integrations; training code should prefer backend-native
/// tensor operations when available.
pub trait R2lTensor: Clone + Send + Sync + Debug + 'static {
    /// Returns the tensor values as a flat vector.
    fn to_vec(&self) -> Result<Vec<f32>>;

    /// Returns the tensor shape.
    fn to_shape(&self) -> Vec<usize>;

    /// Returns the tensors vec and shape
    fn to_vec_and_shape(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        let vec = self.to_vec()?;
        let shape = self.to_shape();
        Ok((vec, shape))
    }

    /// Creates a tensor by copying flat values into `shape`.
    fn from_slice_and_shape(data: &[f32], shape: Vec<usize>) -> Result<Self>;

    /// Constructs a new tensor based on the a vector and shape
    #[must_use]
    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::from_slice_and_shape(&data, shape)
    }

    /// Convert between tensors of different types
    fn convert<S: R2lTensor>(s: &S) -> Result<Self> {
        let (data, shape) = s.to_vec_and_shape()?;
        Self::from_vec_and_shape(data, shape)
    }

    /// Returns the size of the tensor
    fn size(&self) -> usize {
        self.to_shape().iter().product()
    }

    /// Returns true if the tensor is empty
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Elementwise addition.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn add(&self, other: &Self) -> Result<Self>;

    /// Elementwise subtraction.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn sub(&self, other: &Self) -> Result<Self>;

    /// Elementwise multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn mul(&self, other: &Self) -> Result<Self>;

    /// Elementwise exponential.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn exp(&self) -> Result<Self>;

    /// Clamps each element to the inclusive range `[min, max]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn clamp(&self, min: f32, max: f32) -> Result<Self>;

    /// Elementwise minimum between two tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn minimum(&self, other: &Self) -> Result<Self>;

    /// Elementwise negation.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn neg(&self) -> Result<Self>;

    /// Mean reduction over all elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the reduction.
    fn mean(&self) -> Result<Self>;

    /// Elementwise square.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn sqr(&self) -> Result<Self>;

    /// Creates a zero-filled tensor with `shape`.
    #[must_use]
    fn zeros(shape: Vec<usize>) -> Result<Self> {
        let data = vec![0f32; shape.iter().product()];
        Self::from_vec_and_shape(data, shape)
    }

    /// Multiplies every element by `scalar`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn mul_scalar(&self, scalar: f32) -> Result<Self>;

    /// Adds a non-empty slice of equally shaped tensors.
    fn add_multiple(tensors: &[Self]) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyInput {
                operation: "add multiple".into(),
            });
        }
        let shape = tensors[0].to_shape();
        let init = Self::zeros(shape)?;
        tensors.iter().try_fold(init, |acc, elem| acc.add(elem))
    }

    /// Calculates the mean of the tensors.
    fn mean_tensors(tensors: &[Self]) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyInput {
                operation: "mean tensors".into(),
            });
        }
        let sum = Self::add_multiple(tensors)?;
        sum.mul_scalar(1f32 / tensors.len() as f32)
    }

    /// Calculates the elementwise population variance of a non-empty tensor slice.
    fn var_tensors(tensors: &[Self]) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TensorError::EmptyInput {
                operation: "variance".into(),
            });
        }
        let mean = Self::mean_tensors(tensors)?;
        let diffs_sqr = tensors
            .iter()
            .map(|tensor| tensor.sub(&mean)?.sqr())
            .collect::<Result<Vec<_>>>()?;
        let diffs_sqr_sum = Self::add_multiple(&diffs_sqr)?;
        diffs_sqr_sum.mul_scalar(1f32 / tensors.len() as f32)
    }
}

/// Backend-neutral owned tensor payload.
///
/// `TensorData` stores flat `f32` data with an explicit shape. It is useful for
/// simple environments and for converting between backend tensor types
#[derive(Debug, Clone)]
pub struct TensorData {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl TensorData {
    fn ensure_same_shape(&self, other: &Self, operation: &str) -> Result<()> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                operation: operation.into(),
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }
        Ok(())
    }

    /// Creates a one-dimensional tensor from a vector.
    #[must_use]
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    /// Creates tensor data with an explicit shape.
    ///
    /// Returns an error if `shape` does not describe exactly `data.len()` values.
    #[must_use]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected = shape.iter().product();
        if expected != data.len() {
            return Err(TensorError::InvalidShape {
                shape,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Consumes the tensor data and returns its flat values.
    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
}

impl R2lTensor for TensorData {
    fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }

    fn to_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn from_slice_and_shape(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        Self::new(data.to_vec(), shape)
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::new(data, shape)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        self.ensure_same_shape(other, "add")?;
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        self.ensure_same_shape(other, "subtract")?;
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        self.ensure_same_shape(other, "multiply")?;
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self::new(data, self.shape.clone())
    }

    fn exp(&self) -> Result<Self> {
        Self::new(
            self.data.iter().map(|value| value.exp()).collect(),
            self.shape.clone(),
        )
    }

    fn clamp(&self, min: f32, max: f32) -> Result<Self> {
        Self::new(
            self.data
                .iter()
                .map(|value| value.clamp(min, max))
                .collect(),
            self.shape.clone(),
        )
    }

    fn minimum(&self, other: &Self) -> Result<Self> {
        self.ensure_same_shape(other, "minimum")?;
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        Self::new(data, self.shape.clone())
    }

    fn neg(&self) -> Result<Self> {
        Self::new(
            self.data.iter().map(|value| -value).collect(),
            self.shape.clone(),
        )
    }

    fn mean(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TensorError::EmptyInput {
                operation: "mean".into(),
            });
        }
        let mean = self.data.iter().sum::<f32>() / self.data.len() as f32;
        Ok(Self::from_vec(vec![mean]))
    }

    fn sqr(&self) -> Result<Self> {
        Self::new(
            self.data.iter().map(|value| value * value).collect(),
            self.shape.clone(),
        )
    }

    fn zeros(shape: Vec<usize>) -> Result<Self> {
        let len = shape.iter().product();
        Self::new(vec![0.0; len], shape)
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        Self::new(
            self.data.iter().map(|value| value * scalar).collect(),
            self.shape.clone(),
        )
    }
}
