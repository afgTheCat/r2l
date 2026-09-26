use crate::Shape;

#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::{error::TensorError, utils::slice_mean};

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
    fn from_slice_and_shape(data: &[f32], shape: impl Into<Shape>) -> Result<Self>;

    /// Creates a tensor from owned flat values and a shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the values and shape cannot form a backend tensor.
    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        Self::from_slice_and_shape(&data, shape)
    }

    /// Creates values on the same device as `like`, without connecting them to its gradient graph.
    ///
    /// # Errors
    /// Returns an error if values and dimensions are incompatible or allocation fails.
    fn from_vec_like(data: Vec<f32>, shape: impl Into<Shape>, like: &Self) -> Result<Self>;

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

    /// Adds `scalar` to every element.
    ///
    /// # Errors
    /// Returns an error if the backend cannot perform the operation.
    fn add_scalar(&self, scalar: f32) -> Result<Self>;

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

    /// Concatenates a nonempty sequence of tensors along an axis.
    ///
    /// # Errors
    /// Returns an error if ranks or non-concatenated dimensions differ.
    fn cat(tensors: &[Self], dim: usize) -> Result<Self>;

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
    shape: Shape,
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
        let expected = shape.num_elements();
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

    fn to_shape(&self) -> Shape {
        self.shape.clone()
    }

    fn from_slice_and_shape(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        Self::new(data.to_vec(), shape)
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        Self::new(data, shape)
    }

    fn from_vec_like(data: Vec<f32>, shape: impl Into<Shape>, _like: &Self) -> Result<Self> {
        Self::new(data, shape)
    }

    fn add_scalar(&self, scalar: f32) -> Result<Self> {
        Self::new(
            self.data.iter().map(|value| value + scalar).collect(),
            self.shape.clone(),
        )
    }

    fn sum_dim(&self, dim: usize) -> Result<Self> {
        validate_axis(&self.shape, dim)?;
        let mut shape = self.shape.to_vec();
        shape[dim] = 1;
        let shape = Shape::from(shape);
        let stride: usize = self.shape[dim + 1..].iter().product();
        let outer: usize = self.shape[..dim].iter().product();
        let width = self.shape[dim];
        let mut data = Vec::with_capacity(shape.num_elements());
        for group in 0..outer {
            for offset in 0..stride {
                data.push(
                    (0..width)
                        .map(|k| self.data[(group * width + k) * stride + offset])
                        .sum(),
                );
            }
        }
        Self::new(data, shape)
    }

    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let shape = narrow_shape(&self.shape, dim, start, length)?;
        let stride: usize = self.shape[dim + 1..].iter().product();
        let outer: usize = self.shape[..dim].iter().product();
        let mut data = Vec::with_capacity(shape.num_elements());
        for group in 0..outer {
            let begin = (group * self.shape[dim] + start) * stride;
            data.extend_from_slice(&self.data[begin..begin + length * stride]);
        }
        Self::new(data, shape)
    }

    fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        let shapes: Vec<_> = tensors.iter().map(Self::to_shape).collect();
        let shape = concatenated_shape(&shapes, dim)?;
        let outer: usize = shape[..dim].iter().product();
        let stride: usize = shape[dim + 1..].iter().product();
        let mut data = Vec::with_capacity(shape.num_elements());
        for group in 0..outer {
            for tensor in tensors {
                let width = tensor.shape[dim] * stride;
                data.extend_from_slice(&tensor.data[group * width..(group + 1) * width]);
            }
        }
        Self::new(data, shape)
    }

    fn broadcast_as(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        validate_broadcast(&self.shape, &shape)?;
        let data = (0..shape.num_elements())
            .map(|mut index| {
                let mut source = 0;
                let mut stride = 1;
                for (&src, &dst) in self.shape.iter().zip(shape.iter()).rev() {
                    let coordinate = index % dst;
                    index /= dst;
                    if src != 1 {
                        source += coordinate * stride;
                    }
                    stride *= src;
                }
                self.data[source]
            })
            .collect();
        Self::new(data, shape)
    }

    fn log_sigmoid(&self) -> Result<Self> {
        let data = self
            .data
            .iter()
            .map(|&value| {
                if value >= 0. {
                    -(-value).exp().ln_1p()
                } else {
                    value - value.exp().ln_1p()
                }
            })
            .collect();
        Self::new(data, self.shape.clone())
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
        let mean = slice_mean(&self.data);
        Ok(Self::from_vec(vec![mean]))
    }

    fn sqr(&self) -> Result<Self> {
        Self::new(
            self.data.iter().map(|value| value * value).collect(),
            self.shape.clone(),
        )
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
            left: shape.into(),
            right: indices_shape.into(),
        });
    }
    Ok(())
}

fn validate_axis(shape: &Shape, dim: usize) -> Result<()> {
    if dim >= shape.rank() {
        return Err(TensorError::invalid_argument(
            "axis",
            format!("axis {dim} outside {shape:?}"),
        ));
    }
    Ok(())
}

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
