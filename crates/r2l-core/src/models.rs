use anyhow::Result;

use crate::tensor::R2lTensor;

/// A policy-like object that can choose an action for one observation.
///
/// Actors are the inference-time surface used by samplers. They must be
/// sendable so rollout collection can move them into worker threads.
pub trait Actor: Send + 'static {
    /// Tensor type accepted as observations and returned as actions.
    type Tensor: R2lTensor;

    /// Selects an action for a single observation.
    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor>;
}

/// Trainable action distribution interface used by on-policy algorithms.
///
/// A `Policy` extends [`Actor`] with the quantities needed to compute policy
/// gradient losses and entropy bonuses over a batch.
pub trait Policy: Actor {
    /// Computes log probabilities for batched observation/action pairs.
    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor>;

    /// Returns a representative action standard deviation when available.
    fn std(&self) -> Result<f32>;

    /// Computes the policy entropy for a batch of states.
    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor>;

    /// Resamples exploration noise for policies that use state-independent
    /// noise. Implementations without such noise may keep the default no-op.
    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Component that applies backend-specific optimizer updates.
pub trait LearningModule {
    /// Loss bundle consumed by this module.
    type Losses;

    /// Applies one optimization update from precomputed losses.
    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

/// Batched value-function interface.
pub trait ValueFunction {
    /// Tensor type used for observations and returned values.
    type Tensor: Clone;

    /// Estimates values for a batch of observations.
    fn values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor>;
}
