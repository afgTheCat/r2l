//! Experimental policies over backend-independent tensors and network contracts.

pub mod distributions;
pub mod networks;

pub use distributions::{
    bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical, policy::Policy,
};
pub use networks::Network;
use r2l_core::error::Result;
use r2l_core::models::Actor;

/// Trainable action distribution interface used by on-policy algorithms.
///
/// Observations and actions are rank-two batches. Log probabilities have shape
/// `[batch, 1]`; entropy is the batch mean of the summed action-component entropies.
/// Actions supplied for training must lie in the policy's support.
///
/// A `Policy` extends [`Actor`] with the quantities needed to compute policy
/// gradient losses and entropy bonuses over a batch.
pub trait Policy2: Actor {
    /// Shape of one flattened action, without the batch axis.
    fn action_shape(&self) -> r2l_core::Shape;

    /// Computes log probabilities for batched observation/action pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy cannot evaluate the batch.
    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor>;

    /// Returns a representative action standard deviation when available.
    ///
    /// # Errors
    ///
    /// Returns an error if the standard deviation cannot be computed.
    /// Returns `Ok(None)` when the policy has no meaningful scalar standard deviation.
    fn std(&self) -> Result<Option<f32>>;

    /// Computes the policy entropy for a batch of states.
    ///
    /// # Errors
    ///
    /// Returns an error if the entropy cannot be computed.
    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor>;
}
