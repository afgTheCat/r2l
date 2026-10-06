use crate::{
    error::Result,
    models::{Learner, Policy, ValueFunction},
    on_policy::losses::PolicyValueLosses,
};

/// Learner contract using batched policies and value functions.
///
/// This ties together a train-time policy, an inference-time policy, a value
/// function, and a shared loss bundle. Training and inference use the same
/// tensor type; the backend controls gradient tracking at runtime.
pub trait OnPolicyLearner:
    Learner<Losses = PolicyValueLosses<Self::Tensor>> + ValueFunction
{
    /// Policy type used for rollout/inference.
    type InferencePolicy: Policy<Tensor = Self::Tensor> + Clone;
    /// Policy type used while computing losses.
    type Policy: Policy<Tensor = Self::Tensor>;

    /// Prepares rollout data for loss computation without connecting it to an
    /// earlier gradient graph. Burn enables autodiff; Candle detaches the input.
    fn prepare_learning_tensor(t: &Self::Tensor) -> Self::Tensor;

    /// Creates a `[batch, 1]` learning tensor from scalar data, such as returns.
    ///
    /// # Errors
    ///
    /// Returns an error if the learning backend cannot create the tensor.
    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::Tensor>;

    /// Returns a policy suitable for rollout/inference.
    fn inference_policy(&self) -> Self::InferencePolicy;

    /// Returns the train-time policy.
    fn policy(&self) -> &Self::Policy;

    /// Returns the current policy optimizer learning rate.
    fn policy_learning_rate(&self) -> f64;

    /// Sets the same learning rate for policy and value updates.
    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.set_learning_rates(learning_rate, learning_rate);
    }

    /// Sets learning rates independently for policy and value updates.
    /// Joint optimizers use `policy_learning_rate` for both networks.
    fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64);
}
