use crate::{distributions::Policy, tensor::R2lTensor};
use anyhow::Result;

pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

pub trait ValueFunction {
    type Tensor: Clone;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor>;
}

// I don't think we need this
pub trait ModuleWithValueFunction {
    // The tensor type returned to env
    type InferenceTensor: R2lTensor;
    // The tensor type used internally for learning
    type Tensor: R2lTensor;
    // What we need is an inference policy type (maybe actor?)
    type InferencePolicy: Policy<Tensor = Self::InferenceTensor>;
    // The policy that has autograd
    type Policy: Policy<Tensor = Self::Tensor>;
    // The value function
    type ValueFunction: ValueFunction<Tensor = Self::Tensor>;
    // The losses
    type Losses;

    fn get_inference_policy(&self) -> Self::InferencePolicy;

    fn get_policy(&self) -> &Self::Policy;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;

    fn value_func(&self) -> &Self::ValueFunction;
}
