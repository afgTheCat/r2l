#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

/// Minimal tensor contract shared by environments, policies, buffers, and agents.
///
/// Implementors should be cheap enough to clone for rollout storage and safe to
/// move across worker threads. `to_vec` is mainly for inspection, logging, and
/// simple environment integrations; training code should prefer backend-native
/// tensor operations when available.
pub trait R2lTensor: Clone + Send + Sync + Debug + 'static {
    /// Returns the tensor values as a flat vector.
    fn to_vec(&self) -> Vec<f32>;
}

/// Elementwise and reduction operations required by the built-in on-policy
/// algorithms.
///
/// Backends implement this trait for the tensor type used during learning. The
/// methods return `Result` so backend-specific shape or device errors can be
/// propagated without constraining the core crate to one tensor library.
pub trait R2lTensorMath: R2lTensor {
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
}
