use anyhow::Result;

use crate::{
    distributions::Policy,
    losses::PolicyValuesLosses,
    tensor::{R2lTensor, R2lTensorMath},
};

pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

pub trait ValueFunction {
    type Tensor: Clone;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor>;
}

pub trait OnPolicyLearningModule:
    LearningModule<Losses: PolicyValuesLosses<Self::LearningTensor>>
    + ValueFunction<Tensor = Self::LearningTensor>
{
    type InferenceTensor: R2lTensor;
    type LearningTensor: R2lTensorMath;

    type InferencePolicy: Policy<Tensor = Self::InferenceTensor>;
    type Policy: Policy<Tensor = Self::LearningTensor>;

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor;

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor;

    fn get_inference_policy(&self) -> Self::InferencePolicy;

    fn get_policy(&self) -> &Self::Policy;
}
