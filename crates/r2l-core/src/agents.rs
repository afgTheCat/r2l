use crate::{distributions::Policy, sampler::buffer::TrajectoryContainer, tensor::R2lTensor};
use anyhow::Result;

pub trait Agent {
    type Tensor: R2lTensor;

    type Policy: Policy<Tensor = Self::Tensor>;

    fn policy(&self) -> Self::Policy;

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(&mut self, buffers: &[C])
    -> Result<()>;

    fn shutdown(&mut self) {}
}
