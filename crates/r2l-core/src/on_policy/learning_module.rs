use crate::{
    models::{LearningModule, Policy, ValueFunction},
    on_policy::losses::FromPolicyValueLosses,
    tensor::{R2lTensor, R2lTensorMath},
};

pub trait OnPolicyLearningModule:
    LearningModule<Losses: FromPolicyValueLosses<Self::LearningTensor>>
    + ValueFunction<Tensor = Self::LearningTensor>
{
    type InferenceTensor: R2lTensor;
    type LearningTensor: R2lTensorMath;

    type InferencePolicy: Policy<Tensor = Self::InferenceTensor>;
    type Policy: Policy<Tensor = Self::LearningTensor>;

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor;

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor;

    fn inference_policy(&self) -> Self::InferencePolicy;

    fn policy(&self) -> &Self::Policy;
}
