use std::sync::{Arc, Mutex};

// I think we should move this to a different crate eventually
use itertools::izip;
use serde::{Deserialize, Serialize};

use crate::Shape;
use crate::{error::TensorError, running_mean::RunningMeanStd, tensor::R2lTensor};

type Result<T> = std::result::Result<T, TensorError>;

/// Controls whether an observation normalizer mutates shared statistics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizerMode {
    /// Update running statistics before normalizing each batch.
    Update,
    /// Normalize using existing statistics without updating them.
    ReadOnly,
}

const EPSILON: f32 = 1e-8;

#[derive(Clone)]
struct RunningMean<T: R2lTensor> {
    rm: RunningMeanStd<T>,
    clip: Option<f32>,
}

impl<T: R2lTensor> RunningMean<T> {
    pub fn update(&mut self, obs: &[T]) -> Result<()> {
        self.rm.update(obs)
    }

    pub fn normalize_in_place(&self, obs: &mut [T]) -> Result<()> {
        let mean = self.rm.mean.to_vec()?;
        let var = self.rm.var.to_vec()?;
        for obs in obs {
            let (data, shape) = obs.to_vec_and_shape()?;
            let normalized = izip!(data, &mean, &var)
                .map(|(val, mean, var)| {
                    let normalized = (val - mean) / (var + EPSILON).sqrt();
                    if let Some(clip) = self.clip {
                        normalized.clamp(-clip, clip)
                    } else {
                        normalized
                    }
                })
                .collect();
            *obs = T::from_vec_and_shape(normalized, shape)?;
        }
        Ok(())
    }
}

struct NormalizerInner<T: R2lTensor>(Arc<Mutex<RunningMean<T>>>);

impl<T: R2lTensor> Clone for NormalizerInner<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Shared observation normalizer with optional clipping, backed by running statistics.
#[derive(Clone)]
pub struct Normalizer<T: R2lTensor> {
    normalizer_mode: NormalizerMode,
    inner: NormalizerInner<T>,
}

/// Serializable snapshot of a normalizer's statistics and optional clipping limit.
#[derive(Clone, Serialize, Deserialize)]
pub struct NormalizerSnapshot {
    normalizer_mode: NormalizerMode,
    obs_shape: Shape,
    mean: Vec<f32>,
    var: Vec<f32>,
    count: f32,
    clip: Option<f32>,
}

impl<T: R2lTensor> Normalizer<T> {
    /// Returns a handle to the same statistics using `normalizer_mode`.
    #[must_use]
    pub fn with_mode(&self, normalizer_mode: NormalizerMode) -> Self {
        Self {
            normalizer_mode,
            inner: self.inner.clone(),
        }
    }

    /// Creates a normalizer from existing statistics, with optional clipping to `[-clip, clip]`.
    pub fn new(normalizer_mode: NormalizerMode, rm: RunningMeanStd<T>, clip: Option<f32>) -> Self {
        let inner = RunningMean { rm, clip };
        Self {
            normalizer_mode,
            inner: NormalizerInner(Arc::new(Mutex::new(inner))),
        }
    }

    /// Creates a normalizer for observations of `shape`.
    ///
    /// # Panics
    ///
    /// Panics if [`R2lTensor::zeros`] cannot create statistics for `shape`.
    pub fn build(
        normalizer_mode: NormalizerMode,
        clip: Option<f32>,
        shape: impl Into<Shape>,
    ) -> Self {
        let rm = RunningMeanStd::new(shape);
        let inner = RunningMean { rm, clip };
        Self {
            normalizer_mode,
            inner: NormalizerInner(Arc::new(Mutex::new(inner))),
        }
    }

    /// Optionally updates statistics, then normalizes `obs` in place, clipping when configured.
    ///
    /// # Panics
    ///
    /// Panics if the shared statistics lock is poisoned.
    ///
    /// # Errors
    ///
    /// Returns an error if the statistics cannot be updated or applied to `obs`.
    pub fn apply_slice_in_place(&self, obs: &mut [T]) -> Result<()> {
        let mut inner = self.inner.0.lock().unwrap();
        match self.normalizer_mode {
            NormalizerMode::ReadOnly => inner.normalize_in_place(obs),
            NormalizerMode::Update => {
                inner.update(obs)?;
                inner.normalize_in_place(obs)
            }
        }
    }

    /// Optionally updates statistics, then normalizes `obs` in place, clipping when configured.
    ///
    /// # Errors
    ///
    /// Returns an error if the statistics cannot be updated or applied to `obs`.
    pub fn apply_tensor_in_place(&self, obs: &mut T) -> Result<()> {
        self.apply_slice_in_place(std::slice::from_mut(obs))
    }

    /// Captures the current statistics in a backend-independent form.
    ///
    /// # Panics
    ///
    /// Panics if the shared statistics lock is poisoned.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend tensor statistics cannot be extracted.
    pub fn snapshot(&self) -> Result<NormalizerSnapshot> {
        let inner = self.inner.0.lock().unwrap();
        let (mean, obs_shape) = inner.rm.mean.to_vec_and_shape()?;
        Ok(NormalizerSnapshot {
            normalizer_mode: self.normalizer_mode,
            obs_shape,
            mean,
            var: inner.rm.var.to_vec()?,
            count: inner.rm.count,
            clip: inner.clip,
        })
    }
}

impl NormalizerSnapshot {
    /// Reconstructs a normalizer from this snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot values cannot be converted into backend tensors.
    pub fn into_normalizer<T: R2lTensor>(self) -> Result<Normalizer<T>> {
        let mean = T::from_vec_and_shape(self.mean, self.obs_shape.clone())?;
        let var = T::from_vec_and_shape(self.var, self.obs_shape)?;
        let rm = RunningMeanStd::build(mean, var, self.count);
        Ok(Normalizer::new(self.normalizer_mode, rm, self.clip))
    }
}
