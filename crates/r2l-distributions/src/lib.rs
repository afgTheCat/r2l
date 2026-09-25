//! Experimental policies over backend-independent tensors and network contracts.

pub mod distributions;
pub mod learning_modules;
pub mod networks;

pub use distributions::{
    bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical, policy::DistributionKind,
};
pub use networks::Network;
use r2l_core::error::Result;
use r2l_core::models::{Actor, Learner};
use r2l_core::on_policy::losses::FromPolicyValueLosses;
use r2l_core::tensor::R2lTensor;

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

/// Value function over rank-two observation batches.
pub trait ValueFunction2 {
    /// Tensor type used for observations and values.
    type Tensor: R2lTensor;

    /// Returns `[batch, 1]` values for `[batch, features]` observations.
    ///
    /// # Errors
    /// Returns an error for incompatible input dimensions or failed evaluation.
    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor>;
}

/// Experimental learner contract using batched policies and value functions.
///
/// This ties together a train-time policy, an inference-time policy, a value
/// function, tensor conversion helpers, and a loss bundle that can be assembled
/// from policy/value loss terms.
pub trait OnPolicyLearner2:
    Learner<Losses: FromPolicyValueLosses<Self::LearningTensor>>
    + ValueFunction2<Tensor = Self::LearningTensor>
{
    /// Tensor type used by rollout actors and environment buffers.
    type InferenceTensor: R2lTensor;
    /// Tensor type used for differentiable learning computations.
    type LearningTensor: R2lTensor;

    /// Policy type used for rollout/inference.
    type InferencePolicy: Policy2<Tensor = Self::InferenceTensor> + Clone;
    /// Policy type used while computing losses.
    type Policy: Policy2<Tensor = Self::LearningTensor>;

    /// Converts an inference tensor into a learning tensor.
    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor;

    /// Creates a `[batch, 1]` learning tensor from scalar data, such as returns.
    ///
    /// # Errors
    ///
    /// Returns an error if the learning backend cannot create the tensor.
    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::LearningTensor>;

    /// Returns a policy suitable for rollout/inference.
    fn inference_policy(&self) -> Self::InferencePolicy;

    /// Returns the train-time policy.
    fn policy(&self) -> &Self::Policy;

    /// Sets the learning rate used by future updates.
    fn set_learning_rate(&mut self, learning_rate: f64);
}
