use crate::{distributions::Policy, utils::rollout_buffer::RolloutBuffer};
use anyhow::Result;

pub trait Agent {
    // The distribution
    type Dist: Policy;

    /// Retriesve the underlying distribution. This should be inference tbh.
    fn distribution(&self) -> Self::Dist;

    /// Instruments learnging with the rollout buffers collected
    fn learn(&mut self, rollouts: Vec<RolloutBuffer<TensorOfAgent<Self>>>) -> Result<()>;
}

pub type TensorOfAgent<A> = <<A as Agent>::Dist as Policy>::Tensor;
