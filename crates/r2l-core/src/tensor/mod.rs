#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

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
    fn to_vec(&self) -> Vec<f32>;

    /// Returns the tensor shape.
    fn to_shape(&self) -> Vec<usize>;

    /// Returns the tensors vec and shape
    fn to_vec_and_shape(&self) -> (Vec<f32>, Vec<usize>) {
        let vec = self.to_vec();
        let shape = self.to_shape();
        (vec, shape)
    }

    fn from_slice_and_shape(data: &[f32], shape: Vec<usize>) -> Self;

    /// Constructs a new tensor based on the a vector and shape
    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::from_slice_and_shape(&data, shape)
    }

    /// Convert between tensors of different types
    fn convert<S: R2lTensor>(s: &S) -> Self {
        let (data, shape) = s.to_vec_and_shape();
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
    fn add(&self, other: &Self) -> anyhow::Result<Self>;

    /// Elementwise subtraction.
    fn sub(&self, other: &Self) -> anyhow::Result<Self>;

    /// Elementwise multiplication.
    fn mul(&self, other: &Self) -> anyhow::Result<Self>;

    /// Elementwise exponential.
    fn exp(&self) -> anyhow::Result<Self>;

    /// Clamps each element to the inclusive range `[min, max]`.
    fn clamp(&self, min: f32, max: f32) -> anyhow::Result<Self>;

    /// Elementwise minimum between two tensors.
    fn minimum(&self, other: &Self) -> anyhow::Result<Self>;

    /// Elementwise negation.
    fn neg(&self) -> anyhow::Result<Self>;

    /// Mean reduction over all elements.
    fn mean(&self) -> anyhow::Result<Self>;

    /// Elementwise square.
    fn sqr(&self) -> anyhow::Result<Self>;

    fn zeros(shape: Vec<usize>) -> Self {
        let data = vec![0f32; shape.iter().product()];
        Self::from_vec_and_shape(data, shape)
    }

    fn mul_scalar(&self, scalar: f32) -> anyhow::Result<Self>;

    fn add_multiple(tensors: &[Self]) -> Self {
        assert!(!tensors.is_empty());
        let shape = tensors[0].to_shape();
        let init = Self::zeros(shape);
        tensors
            .iter()
            .fold(init, |acc, elem| acc.add(elem).unwrap())
    }

    /// Calculates the mean of the tensors.
    fn mean_tensors(tensors: &[Self]) -> Self {
        assert!(!tensors.is_empty());
        let sum = Self::add_multiple(tensors);
        sum.mul_scalar(1f32 / tensors.len() as f32).unwrap()
    }

    fn var_tensors(tensors: &[Self]) -> Self {
        let mean = Self::mean_tensors(tensors);
        let diffs_sqr = tensors
            .iter()
            .map(|t| t.sub(&mean).unwrap().sqr().unwrap())
            .collect::<Vec<_>>();
        let diffs_sqr_sum = Self::add_multiple(&diffs_sqr);
        diffs_sqr_sum
            .mul_scalar(1f32 / tensors.len() as f32)
            .unwrap()
    }
}

/// Backend-neutral owned tensor payload.
///
/// `TensorData` stores flat `f32` data with an explicit shape. It is useful for
/// simple environments and for converting between backend tensor types
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Flat row-major tensor values.
    pub data: Vec<f32>,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
}

impl TensorData {
    /// Creates a one-dimensional tensor from a vector.
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    /// Creates tensor data with an explicit shape.
    ///
    /// In debug builds, this checks that `shape.iter().product()` matches the
    /// number of values.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert!(shape.iter().product::<usize>() == data.len());
        Self { data, shape }
    }

    /// Consumes the tensor data and returns its flat values.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
}

impl R2lTensor for TensorData {
    fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn to_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn from_slice_and_shape(data: &[f32], shape: Vec<usize>) -> Self {
        Self {
            data: data.to_vec(),
            shape,
        }
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    fn add(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(self.shape == other.shape, "shape mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self::new(data, self.shape.clone()))
    }

    fn sub(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(self.shape == other.shape, "shape mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Self::new(data, self.shape.clone()))
    }

    fn mul(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(self.shape == other.shape, "shape mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Self::new(data, self.shape.clone()))
    }

    fn exp(&self) -> anyhow::Result<Self> {
        Ok(Self::new(
            self.data.iter().map(|value| value.exp()).collect(),
            self.shape.clone(),
        ))
    }

    fn clamp(&self, min: f32, max: f32) -> anyhow::Result<Self> {
        Ok(Self::new(
            self.data
                .iter()
                .map(|value| value.clamp(min, max))
                .collect(),
            self.shape.clone(),
        ))
    }

    fn minimum(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(self.shape == other.shape, "shape mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        Ok(Self::new(data, self.shape.clone()))
    }

    fn neg(&self) -> anyhow::Result<Self> {
        Ok(Self::new(
            self.data.iter().map(|value| -value).collect(),
            self.shape.clone(),
        ))
    }

    fn mean(&self) -> anyhow::Result<Self> {
        let mean = self.data.iter().sum::<f32>() / self.data.len() as f32;
        Ok(Self::from_vec(vec![mean]))
    }

    fn sqr(&self) -> anyhow::Result<Self> {
        Ok(Self::new(
            self.data.iter().map(|value| value * value).collect(),
            self.shape.clone(),
        ))
    }

    fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    fn mul_scalar(&self, scalar: f32) -> anyhow::Result<Self> {
        Ok(Self::new(
            self.data.iter().map(|value| value * scalar).collect(),
            self.shape.clone(),
        ))
    }
}
