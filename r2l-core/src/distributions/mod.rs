use anyhow::Result;
use std::fmt::Debug;

// TODO: Decoding will need a context to store what device we want the tensors to be decoded to
// The phylosophy behind this should be that distributions are stateless, therfore cloenable and
// 'static and self contained. Will see if we can stick to this, but it is the agent that has the
// liberty to not be stateless and such
pub trait Distribution: Sync + Debug {
    type Tensor: Clone;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor>;
    fn log_probs(&self, states: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor>;
    fn std(&self) -> Result<f32>;
    fn entropy(&self) -> Result<Self::Tensor>;
    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}
