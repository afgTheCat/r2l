use crate::{
    models::{LearningModule, Policy, ValueFunction},
    on_policy::losses::FromPolicyValueLosses,
    tensor::{R2lTensor, R2lTensorMath},
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
    type LearningTensor: R2lTensorMath;

    /// Policy type used for rollout/inference.
    type InferencePolicy: Policy<Tensor = Self::InferenceTensor>;
    /// Policy type used while computing losses.
    type Policy: Policy<Tensor = Self::LearningTensor>;

    /// Converts an inference tensor into a learning tensor.
    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor;

    /// Creates a learning tensor from flat scalar data.
    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor;

    /// Returns a policy suitable for rollout/inference.
    fn inference_policy(&self) -> Self::InferencePolicy;

    /// Returns the train-time policy.
    fn policy(&self) -> &Self::Policy;
}
