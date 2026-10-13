use crate::Shape;

#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

use itertools::izip;
use serde::{Deserialize, Serialize};

use crate::error::TensorError;

type Result<T> = std::result::Result<T, TensorError>;

// NOTE: we might want to add int_vec_and_shape method for less allocations, since to_vec_and_shape
// usually clones the inner vector. Would be useful in ActorWrapper.
//
/// Tensor data and arithmetic shared by environments, buffers, and normalization.
/// Backend learning and distribution operations are provided by extension traits.
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
    fn to_shape(&self) -> Shape;

    /// Returns the tensor values and shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend values cannot be extracted.
    fn to_vec_and_shape(&self) -> Result<(Vec<f32>, Shape)> {
        let vec = self.to_vec()?;
        let shape = self.to_shape();
        Ok((vec, shape))
    }

    /// Creates a tensor by copying flat values into `shape`.
    ///
    /// # Errors
    ///
    /// Returns an error if the values and shape cannot form a backend tensor.
    fn from_slice_and_shape(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        Self::from_vec_and_shape(data.to_vec(), shape)
    }

    /// Creates a tensor from owned flat values and a shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the values and shape cannot form a backend tensor.
    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self>;

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
        self.to_shape().num_elements()
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

    /// Elementwise square.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn sqr(&self) -> Result<Self> {
        self.mul(self)
    }

    /// Creates a zero-filled tensor with `shape`.
    ///
    /// # Panics
    ///
    /// Panics if `shape` is invalid, its rank is incompatible with the backend
    /// tensor type, or the backend cannot construct the tensor.
    fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let len = shape.num_elements();
        let data = vec![0f32; len];
        Self::from_vec_and_shape(data, shape).expect("failed to create zero-filled tensor")
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
        tensors
            .iter()
            .skip(1)
            .try_fold(tensors[0].clone(), |acc, elem| acc.add(elem))
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
        let (first, rest) = tensors
            .split_first()
            .ok_or_else(|| TensorError::EmptyInput {
                operation: "variance".into(),
            })?;
        let mean = Self::mean_tensors(tensors)?;
        let diffs_sqr_sum = rest
            .iter()
            .try_fold(first.sub(&mean)?.sqr()?, |sum, tensor| {
                sum.add(&tensor.sub(&mean)?.sqr()?)
            })?;
        diffs_sqr_sum.mul_scalar(1f32 / tensors.len() as f32)
    }
}

/// Backend operations shared by distributions and on-policy learning.
/// Operations must preserve gradient connections unless explicitly detaching values.
pub trait TensorOps: R2lTensor {
    /// Returns the same values without a connection to the gradient graph.
    #[must_use]
    fn detach(&self) -> Self;

    /// Elementwise exponential.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn exp(&self) -> Result<Self>;

    /// Elementwise negation.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the operation.
    fn neg(&self) -> Result<Self> {
        self.mul_scalar(-1.0)
    }

    /// Mean reduction over all elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot perform the reduction.
    fn mean(&self) -> Result<Self>;

    /// Adds `scalar` to every element.
    ///
    /// # Errors
    /// Returns an error if the backend cannot perform the operation.
    fn add_scalar(&self, scalar: f32) -> Result<Self>;

    /// Concatenates a nonempty sequence of tensors along an axis.
    ///
    /// # Errors
    /// Returns an error if ranks or non-concatenated dimensions differ.
    fn cat(tensors: &[Self], dim: usize) -> Result<Self>;
}

/// Additional tensor operations required by built-in on-policy objectives.
pub trait OnPolicyTensor: TensorOps {
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
}

/// Additional tensor operations required by built-in action distributions.
pub trait DistributionTensor: TensorOps {
    /// Creates values on the same device as `like`, without connecting them to its gradient graph.
    ///
    /// # Errors
    /// Returns an error if values and dimensions are incompatible or allocation fails.
    fn from_vec_like(data: Vec<f32>, shape: impl Into<Shape>, like: &Self) -> Result<Self>;

    /// Multiplies rank-two matrices, preserving gradients through both operands.
    ///
    /// # Errors
    /// Returns an error unless the inner dimensions match.
    fn matmul(&self, other: &Self) -> Result<Self>;

    /// Elementwise natural logarithm.
    ///
    /// # Errors
    /// Returns an error if the backend cannot evaluate the logarithm.
    fn log(&self) -> Result<Self>;

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

    /// Sums an axis, retaining it with length one.
    ///
    /// # Errors
    /// Returns an error for an invalid axis or failed reduction.
    fn sum_dim(&self, dim: usize) -> Result<Self>;

    /// Selects a contiguous range along an axis, preserving rank.
    ///
    /// # Errors
    /// Returns an error if the axis or range is invalid.
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self>;

    /// Expands singleton axes to a target shape of the same rank.
    ///
    /// # Errors
    /// Returns an error if dimensions cannot be broadcast.
    fn broadcast_as(&self, shape: impl Into<Shape>) -> Result<Self>;

    /// Computes elementwise log-sigmoid without overflowing for large finite logits.
    ///
    /// # Errors
    /// Returns an error if the backend cannot perform the operation.
    fn log_sigmoid(&self) -> Result<Self>;
}

/// Built-in tensor backed by a flat vector and an explicit shape.
///
/// `VecTensor` is useful for environments and for converting between backend
/// tensor types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecTensor {
    data: Vec<f32>,
    shape: Shape,
}

impl VecTensor {
    /// Clamps each element between the corresponding values in `min` and `max`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor shapes differ or the result cannot be constructed.
    pub fn clamp(&self, min: &Self, max: &Self) -> Result<Self> {
        self.shape.ensure_equal(&min.shape, "clamp minimum")?;
        self.shape.ensure_equal(&max.shape, "clamp maximum")?;
        let data = izip!(&self.data, &min.data, &max.data)
            .map(|(value, min, max)| value.clamp(*min, *max))
            .collect();
        Self::new(data, self.shape.clone())
    }

    fn map(&self, f: impl FnMut(f32) -> f32) -> Result<Self> {
        Self::new(
            self.data.iter().copied().map(f).collect(),
            self.shape.clone(),
        )
    }

    fn zip_map(
        &self,
        other: &Self,
        operation: &str,
        mut f: impl FnMut(f32, f32) -> f32,
    ) -> Result<Self> {
        self.shape.ensure_equal(&other.shape, operation)?;
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| f(a, b))
            .collect();
        Self::new(data, self.shape.clone())
    }

    /// Creates a one-dimensional tensor from a vector.
    #[must_use]
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = Shape::from([data.len()]);
        Self { data, shape }
    }

    /// Creates tensor data with an explicit shape.
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` does not describe exactly `data.len()` values.
    pub fn new(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate_len(data.len())?;
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

    fn to_shape(&self) -> Shape {
        self.shape.clone()
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        Self::new(data, shape)
    }

    fn add(&self, other: &Self) -> Result<Self> {
        self.zip_map(other, "add", |a, b| a + b)
    }

    fn sub(&self, other: &Self) -> Result<Self> {
        self.zip_map(other, "subtract", |a, b| a - b)
    }

    fn mul(&self, other: &Self) -> Result<Self> {
        self.zip_map(other, "multiply", |a, b| a * b)
    }

    fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        self.map(|value| value * scalar)
    }
}

#[cfg(any(feature = "candle", feature = "burn"))]
fn matrix_dimensions(left: &Shape, right: &Shape) -> Result<[usize; 3]> {
    match (left.dims(), right.dims()) {
        ([rows, inner], [other_inner, columns]) if inner == other_inner => {
            Ok([*rows, *inner, *columns])
        }
        _ => Err(TensorError::ShapeMismatch {
            operation: "matrix multiplication".into(),
            left: left.clone(),
            right: right.clone(),
        }),
    }
}

#[cfg(any(feature = "candle", feature = "burn"))]
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
            left: shape.into(),
            right: indices_shape.into(),
        });
    }
    Ok(())
}

#[cfg(any(feature = "candle", feature = "burn"))]
fn validate_axis(shape: &Shape, dim: usize) -> Result<()> {
    if dim >= shape.rank() {
        return Err(TensorError::invalid_argument(
            "axis",
            format!("axis {dim} outside {shape:?}"),
        ));
    }
    Ok(())
}

#[cfg(any(feature = "candle", feature = "burn"))]
fn narrow_shape(shape: &Shape, dim: usize, start: usize, length: usize) -> Result<Shape> {
    validate_axis(shape, dim)?;
    if start.checked_add(length).is_none_or(|end| end > shape[dim]) {
        return Err(TensorError::invalid_argument(
            "narrow",
            "range exceeds axis length",
        ));
    }
    let mut result = shape.to_vec();
    result[dim] = length;
    Ok(result.into())
}

#[cfg(any(feature = "candle", feature = "burn"))]
fn concatenated_shape(shapes: &[Shape], dim: usize) -> Result<Shape> {
    let first = shapes.first().ok_or_else(|| TensorError::EmptyInput {
        operation: "concatenate".into(),
    })?;
    validate_axis(first, dim)?;
    let mut result = first.to_vec();
    result[dim] = 0;
    for shape in shapes {
        if shape.rank() != first.rank()
            || shape
                .iter()
                .zip(first.iter())
                .enumerate()
                .any(|(axis, (a, b))| axis != dim && a != b)
        {
            return Err(TensorError::ShapeMismatch {
                operation: "concatenate".into(),
                left: first.clone(),
                right: shape.clone(),
            });
        }
        result[dim] = result[dim].checked_add(shape[dim]).ok_or_else(|| {
            TensorError::invalid_argument("concatenate", "axis length overflows usize")
        })?;
    }
    Ok(result.into())
}

#[cfg(any(feature = "candle", feature = "burn"))]
fn validate_broadcast(source: &Shape, target: &Shape) -> Result<()> {
    if source.rank() != target.rank()
        || source
            .iter()
            .zip(target.iter())
            .any(|(&a, &b)| a != 1 && a != b)
    {
        return Err(TensorError::ShapeMismatch {
            operation: "broadcast".into(),
            left: source.clone(),
            right: target.clone(),
        });
    }
    Ok(())
}
