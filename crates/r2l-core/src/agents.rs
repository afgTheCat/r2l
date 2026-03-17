use std::borrow::Cow;

// use crate::sampler3::buffer_stack::BufferStack3;
// use crate::sampler3::buffers::{Buffer, BufferStack};
use crate::{
    distributions::Policy, sampler5::buffer::TrajectoryContainer, tensor::R2lTensor,
    utils::rollout_buffer::RolloutBuffer,
};
use anyhow::Result;
use candle_core::Tensor;

pub trait Agent {
    /// The policy
    type Policy: Policy;

    /// Retriesve the underlying distribution. This should be inference tbh.
    fn policy(&self) -> Self::Policy;

    /// Instruments learnging with the rollout buffers collected
    fn learn(&mut self, rollouts: Vec<RolloutBuffer<TensorOfAgent<Self>>>) -> Result<()>;
}

pub type TensorOfAgent<A> = <<A as Agent>::Policy as Policy>::Tensor;

pub trait Agent5 {
    type Tensor: R2lTensor;

    type Policy: Policy<Tensor = Self::Tensor>;

    fn policy(&self) -> Self::Policy;

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(&self, buffers: &[C]) -> Result<()>;
}
