use crate::sampler3::buffers::BufferStack;
use crate::{distributions::Policy, sampler2::Buffer, utils::rollout_buffer::RolloutBuffer};
use anyhow::Result;

pub trait Agent {
    /// The policy
    type Policy: Policy;

    /// Retriesve the underlying distribution. This should be inference tbh.
    fn policy(&self) -> Self::Policy;

    /// Instruments learnging with the rollout buffers collected
    fn learn(&mut self, rollouts: Vec<RolloutBuffer<TensorOfAgent<Self>>>) -> Result<()>;
}

pub type TensorOfAgent<A> = <<A as Agent>::Policy as Policy>::Tensor;

pub trait Agent2 {
    /// The policy
    type Policy: Policy;

    /// Retriesve the underlying distribution. This should be inference tbh.
    fn policy2(&self) -> Self::Policy;

    /// Instruments learnging with the rollout buffers collected
    fn learn2<B: Buffer>(&mut self, buffers: &[B]) -> Result<()>
    where
        <B as Buffer>::Tensor: Into<<Self::Policy as Policy>::Tensor>;
}

pub type TensorOfAgent2<A> = <<A as Agent2>::Policy as Policy>::Tensor;

pub trait Agent3 {
    /// The policy
    type Policy: Policy;

    /// Retriesve the underlying distribution. This should be inference tbh.
    fn policy3(&self) -> Self::Policy;

    /// Instruments learnging with the rollout buffers collected
    fn learn3<B: Buffer>(&mut self, buffers: BufferStack<B>) -> Result<()>
    where
        <Self::Policy as Policy>::Tensor: From<<B as Buffer>::Tensor>;
}

pub type TensorOfAgent3<A> = <<A as Agent3>::Policy as Policy>::Tensor;
