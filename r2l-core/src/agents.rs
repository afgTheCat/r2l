use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::Result;

pub trait Agent {
    // The distribution
    type Dist: Distribution;

    /// Retriesve the underlying distribution
    fn distribution(&self) -> &Self::Dist;

    /// Instruments learnging with the rollout buffers collected
    fn learn(&mut self, rollouts: Vec<RolloutBuffer<TensorOfAgent<Self>>>) -> Result<()>;
}

pub type TensorOfAgent<A> = <<A as Agent>::Dist as Distribution>::Tensor;
