use anyhow::Result;

use crate::tensor::R2lTensor;

pub trait Actor: Send + 'static {
    type Tensor: R2lTensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor>;
}

pub trait Policy: Actor {
    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor>;

    fn std(&self) -> Result<f32>;

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor>;

    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}

pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

pub trait ValueFunction {
    type Tensor: Clone;

    fn values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor>;
}
