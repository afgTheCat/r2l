use crate::{
    error::Result,
    models::{Learner, Policy, ValueFunction},
    on_policy::losses::FromPolicyValueLosses,
    tensor::R2lTensor,
};

/// Learner contract using batched policies and value functions.
///
/// This ties together a train-time policy, an inference-time policy, a value
/// function, tensor conversion helpers, and a loss bundle that can be assembled
/// from policy/value loss terms.
pub trait OnPolicyLearner:
    Learner<Losses: FromPolicyValueLosses<Self::LearningTensor>>
    + ValueFunction<Tensor = Self::LearningTensor>
{
    /// Tensor type used by rollout actors and environment buffers.
    type InferenceTensor: R2lTensor;
    /// Tensor type used for differentiable learning computations.
    type LearningTensor: R2lTensor;

    /// Policy type used for rollout/inference.
    type InferencePolicy: Policy<Tensor = Self::InferenceTensor> + Clone;
    /// Policy type used while computing losses.
    type Policy: Policy<Tensor = Self::LearningTensor>;

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

    /// Sets the same learning rate for policy and value updates.
    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.set_learning_rates(learning_rate, learning_rate);
    }

    /// Sets learning rates independently for policy and value updates.
    /// Joint optimizers use `policy_learning_rate` for both networks.
    fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64);
}
