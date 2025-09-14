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

pub trait Agent2<B: Buffer> {
    /// The policy
    type Policy: Policy;

    /// Retriesve the underlying distribution. This should be inference tbh.
    fn policy2(&self) -> Self::Policy;

    /// Instruments learnging with the rollout buffers collected
    fn learn2(&mut self, buffers: Vec<B>) -> Result<()>;
}
