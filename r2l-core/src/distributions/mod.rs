pub mod categorical_distribution;
pub mod diagonal_distribution;

use candle_core::{Result, Tensor};
use candle_nn::{Sequential, seq};
use categorical_distribution::CategoricalDistribution;
use derive_more::Deref;
use diagonal_distribution::DiagGaussianDistribution;
use enum_dispatch::enum_dispatch;
use std::f32;

#[derive(Deref)]
pub struct ThreadSafeSequential(pub Sequential);

// Just for testing things
impl Default for ThreadSafeSequential {
    fn default() -> Self {
        Self(seq())
    }
}

// SAFETY: ThreadSafeSequential will only contain Linear and Relu layers, both of which are Sync.
unsafe impl Sync for ThreadSafeSequential {}

#[enum_dispatch]
pub trait Distribution: Sync {
    fn get_action(&self, observation: &Tensor) -> Result<(Tensor, Tensor)>;
    fn log_probs(&self, states: &Tensor, actions: &Tensor) -> Result<Tensor>;
    fn std(&self) -> Result<f32>;
    fn entropy(&self) -> Result<Tensor>;
    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}

#[enum_dispatch(Distribution)]
pub enum DistributionKind {
    Categorical(CategoricalDistribution),
    DiagGaussian(DiagGaussianDistribution),
}
