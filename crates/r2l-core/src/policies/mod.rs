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
