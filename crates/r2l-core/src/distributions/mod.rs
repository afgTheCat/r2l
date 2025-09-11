use anyhow::Result;
use std::fmt::Debug;

pub trait Policy: Send + Debug + 'static {
    type Tensor: Clone;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor>;

    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor>;

    fn std(&self) -> Result<f32>;

    fn entropy(&self) -> Result<Self::Tensor>;

    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}
