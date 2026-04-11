use crate::tensor::R2lTensor;
use anyhow::Result;

pub trait Actor: Send + 'static {
    type Tensor: R2lTensor;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor>;
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
