#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

// NOTE: we might want to add int_vec_and_shape method for less allocations, since to_vec_and_shape
// usually clones the inner vector. Would be useful in ActorWrapper.
//
/// Minimal tensor contract shared by environments, policies, buffers, and agents.
///
/// Implementors should be cheap enough to clone for rollout storage and safe to
/// move across worker threads. `to_vec` is mainly for inspection, logging, and
/// simple environment integrations; training code should prefer backend-native
/// tensor operations when available.
pub trait R2lTensor: Clone + Send + Sync + Debug + 'static {
    /// Returns the tensor values as a flat vector.
    fn to_vec(&self) -> Vec<f32>;

    /// Returns the tensor shape
    fn to_shape(&self) -> Vec<usize>;

    /// Returns the tensors vec and shape
    fn to_vec_and_shape(&self) -> (Vec<f32>, Vec<usize>) {
        let vec = self.to_vec();
        let shape = self.to_shape();
        (vec, shape)
    }

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self;

    fn convert<S: R2lTensor>(s: &S) -> Self {
        let (data, shape) = s.to_vec_and_shape();
        Self::from_vec_and_shape(data, shape)
    }

    // TODO: this default impl might be wasteful
    /// Returns the length of the tensor
    fn len(&self) -> usize {
        self.to_vec().len()
    }

    // TODO: this deafult might be wasteful
    /// Returns true if the tensor is empty
    fn is_empty(&self) -> bool {
        self.to_vec().is_empty()
    }
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

// TODO: we need this to be removed
pub trait RunningMeanTensor: R2lTensorMath {
    fn zeros(shape: Vec<usize>) -> Self;
    fn batch_mean(&self) -> anyhow::Result<Self>;
    fn biased_var(&self) -> anyhow::Result<Self>;
    fn batch_count(&self) -> anyhow::Result<f32>;
    fn mul_scalar(&self, scalar: f32) -> anyhow::Result<Self>;
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

    fn from_vec_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

impl R2lTensorMath for TensorData {
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
}

impl RunningMeanTensor for TensorData {
    fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    fn batch_mean(&self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !self.shape.is_empty(),
            "running mean update expects a batch dimension"
        );
        let batch_count = self.shape[0];
        let feature_shape = self.shape[1..].to_vec();
        let feature_size = feature_shape.iter().product::<usize>();
        anyhow::ensure!(
            batch_count * feature_size == self.data.len(),
            "batch tensor shape does not match data length"
        );

        let mut batch_mean = vec![0.0; feature_size];
        for sample in self.data.chunks_exact(feature_size) {
            for (mean, value) in batch_mean.iter_mut().zip(sample.iter()) {
                *mean += *value;
            }
        }
        for mean in &mut batch_mean {
            *mean /= batch_count as f32;
        }
        Ok(Self::from_vec_and_shape(batch_mean, feature_shape))
    }

    fn biased_var(&self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !self.shape.is_empty(),
            "running mean update expects a batch dimension"
        );
        let batch_count = self.shape[0];
        let feature_shape = self.shape[1..].to_vec();
        let feature_size = feature_shape.iter().product::<usize>();
        anyhow::ensure!(
            batch_count * feature_size == self.data.len(),
            "batch tensor shape does not match data length"
        );

        let batch_mean = self.batch_mean()?;
        let batch_mean = batch_mean.data;
        let mut batch_var = vec![0.0; feature_size];
        for sample in self.data.chunks_exact(feature_size) {
            for ((var, value), mean) in batch_var
                .iter_mut()
                .zip(sample.iter())
                .zip(batch_mean.iter())
            {
                let delta = *value - *mean;
                *var += delta * delta;
            }
        }
        for var in &mut batch_var {
            *var /= batch_count as f32;
        }
        Ok(Self::from_vec_and_shape(batch_var, feature_shape))
    }

    fn batch_count(&self) -> anyhow::Result<f32> {
        anyhow::ensure!(
            !self.shape.is_empty(),
            "running mean update expects a batch dimension"
        );
        Ok(self.shape[0] as f32)
    }

    fn mul_scalar(&self, scalar: f32) -> anyhow::Result<Self> {
        Ok(Self::new(
            self.data.iter().map(|value| value * scalar).collect(),
            self.shape.clone(),
        ))
    }
}
