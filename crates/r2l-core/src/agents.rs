use crate::{distributions::Policy, sampler5::buffer::TrajectoryContainer, tensor::R2lTensor};
use anyhow::Result;

pub trait Agent5 {
    type Tensor: R2lTensor;

    type Policy: Policy<Tensor = Self::Tensor>;

    fn policy(&self) -> Self::Policy;

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(&mut self, buffers: &[C])
    -> Result<()>;
}
