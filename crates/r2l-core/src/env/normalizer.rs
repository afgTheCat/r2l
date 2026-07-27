use std::sync::{Arc, Mutex};

// I think we should move this to a different crate eventually
use itertools::izip;

use crate::{running_mean::RunningMeanStd, tensor::R2lTensor};

/// Controls whether a normalized sampler mutates shared normalization stats.
#[derive(Debug, Clone, Copy)]
pub enum NormalizerMode {
    /// Update running statistics before normalizing each batch.
    Update,
    /// Normalize using existing statistics without updating them.
    ReadOnly,
}

const EPSILON: f32 = 1e-8;

#[derive(Clone)]
pub struct ClippedNormalizerInner<T: R2lTensor> {
    rm: RunningMeanStd<T>,
    clip: f32,
}

impl<T: R2lTensor> ClippedNormalizerInner<T> {
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

/// Shared, clipped observation normalizer backed by running statistics.
#[derive(Clone)]
pub struct ClippedNormalizer<T: R2lTensor> {
    normalizer_mode: NormalizerMode,
    inner: Arc<Mutex<ClippedNormalizerInner<T>>>,
}

impl<T: R2lTensor> ClippedNormalizer<T> {
    /// Creates a normalizer for observations of `shape`.
    pub fn new(normalizer_mode: NormalizerMode, clip: f32, shape: Vec<usize>) -> Self {
        let rm = RunningMeanStd::new(shape);
        let inner = ClippedNormalizerInner { clip, rm };
        Self {
            normalizer_mode,
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Clones this normalizer while changing whether it updates shared statistics.
    pub fn with_mode(&self, normalizer_mode: NormalizerMode) -> Self {
        Self {
            normalizer_mode,
            inner: self.inner.clone(),
        }
    }

    /// Optionally updates statistics, then normalizes and clips `obs` in place.
    pub fn apply_in_place(&self, obs: &mut [T]) {
        let mut inner = self.inner.lock().unwrap();
        match self.normalizer_mode {
            NormalizerMode::ReadOnly => inner.normalize_in_place(obs),
            NormalizerMode::Update => {
                inner.update(obs);
                inner.normalize_in_place(obs);
            }
        }
    }
}
