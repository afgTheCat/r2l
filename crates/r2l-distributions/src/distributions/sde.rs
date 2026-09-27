//! Generalized state-dependent exploration with one noise matrix per actor clone.

use std::{num::NonZeroUsize, sync::Mutex};

use r2l_core::{
    error::{Error, Result},
    rng::with_rng,
    tensor::R2lTensor,
};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// Generalized state-dependent exploration for continuous actions.
///
/// Noise uses the policy's last hidden features and a learned scale for every
/// feature/action pair. Each rollout worker receives an independent matrix.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SdeConfig {
    /// Refresh noise after this many actions, in addition to each rollout boundary.
    /// `None` keeps the same matrix throughout a rollout, including episode resets.
    pub resample_every: Option<NonZeroUsize>,
}

#[derive(Debug)]
struct Noise<T> {
    matrix: Option<T>,
    actions: usize,
}

#[derive(Debug)]
pub(super) struct StateDependentNoise<T> {
    pub config: SdeConfig,
    noise: Mutex<Noise<T>>,
}

impl<T> StateDependentNoise<T> {
    pub fn new(config: SdeConfig) -> Self {
        Self {
            config,
            noise: Mutex::new(Noise {
                matrix: None,
                actions: 0,
            }),
        }
    }
}

// Samplers install a fresh actor clone per worker at each rollout boundary.
// Clones deliberately start without a matrix so workers never share exploration draws.
impl<T> Clone for StateDependentNoise<T> {
    fn clone(&self) -> Self {
        Self::new(self.config)
    }
}

impl<T: R2lTensor> StateDependentNoise<T> {
    pub fn sample(&self, features: &T, log_std: &T) -> Result<T> {
        let mut noise = self.noise.lock().map_err(|_| Error::InvalidState {
            operation: "sample state-dependent noise".into(),
            details: "noise lock was poisoned".into(),
        })?;
        if noise.matrix.is_none()
            || self
                .config
                .resample_every
                .is_some_and(|period| noise.actions >= period.get())
        {
            let values = with_rng(|rng| {
                (0..log_std.size())
                    .map(|_| StandardNormal.sample(rng))
                    .collect()
            });
            let standard_normal = T::from_vec_like(values, log_std.to_shape(), log_std)?;
            noise.matrix = Some(standard_normal.mul(&log_std.exp()?)?);
            noise.actions = 0;
        }
        noise.actions = noise.actions.saturating_add(1);
        Ok(features.matmul(noise.matrix.as_ref().expect("noise was initialized"))?)
    }
}
