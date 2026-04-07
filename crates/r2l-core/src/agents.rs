use crate::{distributions::Actor, sampler::buffer::TrajectoryContainer, tensor::R2lTensor};
use anyhow::Result;

pub trait Agent {
    type Tensor: R2lTensor;

    type Actor: Actor<Tensor = Self::Tensor>;

    fn actor(&self) -> Self::Actor;

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(&mut self, buffers: &[C])
    -> Result<()>;

    fn shutdown(&mut self) {}
}
