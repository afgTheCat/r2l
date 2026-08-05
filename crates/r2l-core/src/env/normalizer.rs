use std::sync::{Arc, Mutex};

// I think we should move this to a different crate eventually
use itertools::izip;
use serde::{Deserialize, Serialize};

use crate::{running_mean::RunningMeanStd, tensor::R2lTensor};

/// Controls whether an observation normalizer mutates shared statistics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizerMode {
    /// Update running statistics before normalizing each batch.
    Update,
    /// Normalize using existing statistics without updating them.
    ReadOnly,
}

const EPSILON: f32 = 1e-8;

#[derive(Clone, Serialize, Deserialize)]
pub struct ClippedRunningMean<T: R2lTensor> {
    pub rm: RunningMeanStd<T>,
    pub clip: f32,
}

impl<T: R2lTensor> ClippedRunningMean<T> {
    pub fn update(&mut self, obs: &[T]) {
        self.rm.update(obs);
    }

    pub fn normalize_in_place(&self, obs: &mut [T]) {
        let mean = self.rm.mean.to_vec();
        let var = self.rm.var.to_vec();
        for obs in obs {
            let (data, shape) = obs.to_vec_and_shape();
            let normalized = izip!(data, &mean, &var)
                .map(|(val, mean, var)| {
                    ((val - mean) / (var + EPSILON).sqrt()).clamp(-self.clip, self.clip)
                })
                .collect();
            *obs = T::from_vec_and_shape(normalized, shape);
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ClippedNormalizerInner<T: R2lTensor>(pub Arc<Mutex<ClippedRunningMean<T>>>);

impl<T: R2lTensor> Clone for ClippedNormalizerInner<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Shared, clipped observation normalizer backed by running statistics.
#[derive(Clone, Serialize, Deserialize)]
pub struct ClippedNormalizer<T: R2lTensor> {
    pub normalizer_mode: NormalizerMode,
    pub inner: ClippedNormalizerInner<T>,
}

impl<T: R2lTensor> ClippedNormalizer<T> {
    pub fn new(normalizer_mode: NormalizerMode, rm: RunningMeanStd<T>, clip: f32) -> Self {
        let inner = ClippedRunningMean { clip, rm };
        Self {
            normalizer_mode,
            inner: ClippedNormalizerInner(Arc::new(Mutex::new(inner))),
        }
    }

    /// Creates a normalizer for observations of `shape`.
    pub fn build(normalizer_mode: NormalizerMode, clip: f32, shape: Vec<usize>) -> Self {
        let rm = RunningMeanStd::new(shape);
        let inner = ClippedRunningMean { clip, rm };
        Self {
            normalizer_mode,
            inner: ClippedNormalizerInner(Arc::new(Mutex::new(inner))),
        }
    }

    /// Optionally updates statistics, then normalizes and clips `obs` in place.
    pub fn apply_slice_in_place(&self, obs: &mut [T]) {
        let mut inner = self.inner.0.lock().unwrap();
        match self.normalizer_mode {
            NormalizerMode::ReadOnly => inner.normalize_in_place(obs),
            NormalizerMode::Update => {
                inner.update(obs);
                inner.normalize_in_place(obs);
            }
        }
    }

    pub fn apply_tensor_in_place(&self, obs: &mut T) {
        self.apply_slice_in_place(std::slice::from_mut(obs));
    }
}
