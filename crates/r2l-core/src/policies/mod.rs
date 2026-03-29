use anyhow::Result;

use crate::{distributions::Policy, tensor::R2lTensor};

pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

pub trait ValueFunction {
    type Tensor: Clone;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor>;
}

// This might be super complicated
pub trait ModuleWithValueFunction {
    type Tensor: R2lTensor;
    type InnerTensor: R2lTensor;
    // What we need is an inference policy type (maybe actor?)
    type InferencePolicy: Policy<Tensor = Self::InnerTensor>;
}
