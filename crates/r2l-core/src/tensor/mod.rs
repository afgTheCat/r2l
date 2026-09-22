#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

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
    ///
    /// # Errors
    ///
    /// Returns an error if the backend values cannot be extracted.
    fn to_vec(&self) -> Result<Vec<f32>>;

    /// Returns the tensor shape.
    fn to_shape(&self) -> Vec<usize>;

    /// Returns the tensor values and shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend values cannot be extracted.
    fn to_vec_and_shape(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        let vec = self.to_vec()?;
        let shape = self.to_shape();
        Ok((vec, shape))
    }

    /// Creates a tensor by copying flat values into `shape`.
    ///
    /// # Errors
    ///
    /// Returns an error if the values and shape cannot form a backend tensor.
    fn from_slice_and_shape(data: &[f32], shape: Vec<usize>) -> Result<Self>;

    /// Creates a tensor from owned flat values and a shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the values and shape cannot form a backend tensor.
    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::from_slice_and_shape(&data, shape)
    }

    /// Converts a tensor from another backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the source values cannot be extracted or the destination tensor
    /// cannot be constructed.
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

    /// Selects values along `dim` using category indices stored in `indices`.
    ///
    /// The tensors must have the same rank and match in every dimension except
    /// `dim`. The result has the shape of `indices`. Index values must be
    /// non-negative integers smaller than the size of `dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes are incompatible or the backend cannot
    /// perform the operation.
    fn gather(&self, dim: usize, indices: &Self) -> Result<Self>;

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

    /// Normalizes values into probabilities along `dim`, preserving the shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot perform the operation.
    fn softmax(&self, dim: usize) -> Result<Self>;

    /// Computes log probabilities along `dim`, preserving the shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot perform the operation.
    fn log_softmax(&self, dim: usize) -> Result<Self>;

    /// Creates a zero-filled tensor with `shape`.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot create a tensor with `shape`.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `tensors` is empty or a backend operation fails.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `tensors` is empty or a backend operation fails.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `tensors` is empty or a backend operation fails.
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

/// Built-in tensor backed by a flat vector and an explicit shape.
///
/// `VecTensor` is useful for environments and for converting between backend
/// tensor types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl VecTensor {
    fn normalized_exp(&self, dim: usize, logarithmic: bool) -> Result<Self> {
        let Some(&width) = self.shape.get(dim).filter(|&&width| width > 0) else {
            return Err(TensorError::operation(
                "softmax",
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "invalid softmax dimension",
                ),
            ));
        };
        let stride: usize = self.shape[dim + 1..].iter().product();
        let outer: usize = self.shape[..dim].iter().product();
        let mut data = self.data.clone();
        for batch in 0..outer {
            for offset in 0..stride {
                let start = batch * width * stride + offset;
                let max = (0..width)
                    .map(|index| self.data[start + index * stride])
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum = (0..width)
                    .map(|index| (self.data[start + index * stride] - max).exp())
                    .sum::<f32>();
                let log_sum = sum.ln();
                for index in 0..width {
                    let position = start + index * stride;
                    let shifted = self.data[position] - max;
                    data[position] = if logarithmic {
                        shifted - log_sum
                    } else {
                        shifted.exp() / sum
                    };
                }
            }
        }
        Self::new(data, self.shape.clone())
    }

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
    /// # Errors
    ///
    /// Returns an error if `shape` does not describe exactly `data.len()` values.
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

impl R2lTensor for VecTensor {
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

    fn gather(&self, dim: usize, indices: &Self) -> Result<Self> {
        validate_gather(&self.shape, &indices.shape, dim)?;
        let stride: usize = self.shape[dim + 1..].iter().product();
        let data = indices
            .data
            .iter()
            .enumerate()
            .map(|(position, &index)| {
                if !index.is_finite()
                    || index < 0.0
                    || index.fract() != 0.0
                    || index as usize >= self.shape[dim]
                {
                    return Err(TensorError::operation(
                        "gather",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "invalid gather index",
                        ),
                    ));
                }
                let outer = position / (indices.shape[dim] * stride);
                let offset = position % stride;
                Ok(self.data[(outer * self.shape[dim] + index as usize) * stride + offset])
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(data, indices.shape.clone())
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

    fn softmax(&self, dim: usize) -> Result<Self> {
        self.normalized_exp(dim, false)
    }

    fn log_softmax(&self, dim: usize) -> Result<Self> {
        self.normalized_exp(dim, true)
    }
}

fn validate_gather(shape: &[usize], indices_shape: &[usize], dim: usize) -> Result<()> {
    if dim >= shape.len()
        || shape.len() != indices_shape.len()
        || shape
            .iter()
            .zip(indices_shape)
            .enumerate()
            .any(|(axis, (left, right))| axis != dim && left != right)
    {
        return Err(TensorError::ShapeMismatch {
            operation: "gather".into(),
            left: shape.to_vec(),
            right: indices_shape.to_vec(),
        });
    }
    Ok(())
}
