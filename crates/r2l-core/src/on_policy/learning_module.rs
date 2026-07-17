use crate::{
    models::{LearningModule, Policy, ValueFunction},
    on_policy::losses::FromPolicyValueLosses,
    tensor::R2lTensor,
};

/// Learning module contract required by the built-in on-policy algorithms.
///
/// This ties together a train-time policy, an inference-time policy, a value
/// function, tensor conversion helpers, and a loss bundle that can be assembled
/// from policy/value loss terms.
pub trait OnPolicyLearningModule:
    LearningModule<Losses: FromPolicyValueLosses<Self::LearningTensor>>
    + ValueFunction<Tensor = Self::LearningTensor>
{
    /// Tensor type used by rollout actors and environment buffers.
    type InferenceTensor: R2lTensor;
    /// Tensor type used for differentiable learning computations.
    type LearningTensor: R2lTensor;
    /// Recurrent state produced by the inference policy during rollouts.
    type InferenceState: Clone + Send + Sync + 'static;
    /// Recurrent state consumed by the differentiable policy during learning.
    type LearningState: Clone + Send + Sync + 'static;

    /// Policy type used for rollout/inference.
    type InferencePolicy: Policy<Tensor = Self::InferenceTensor, State = Self::InferenceState>;
    /// Policy type used while computing losses.
    type Policy: Policy<Tensor = Self::LearningTensor, State = Self::LearningState>;

    /// Converts an inference tensor into a learning tensor.
    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor;

    /// Converts a detached inference state into a train-time state.
    fn state_lifter(state: &Self::InferenceState) -> Self::LearningState;

    /// Creates a learning tensor from flat scalar data.
    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor;

    /// Returns a policy suitable for rollout/inference.
    fn inference_policy(&self) -> Self::InferencePolicy;

    /// Returns the train-time policy.
    fn policy(&self) -> &Self::Policy;
}
