use anyhow::Result;

use crate::{buffers::TrajectoryContainer, distributions::Actor, tensor::R2lTensor};

pub trait Agent {
    type Tensor: R2lTensor;

    type Actor: Actor<Tensor = Self::Tensor>;

    fn actor(&self) -> Self::Actor;

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(&mut self, buffers: &[C])
    -> Result<()>;

    fn shutdown(&mut self) {}
}
