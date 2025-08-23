// use super::policies::Policy;
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::{Result, Tensor};

pub trait Agent {
    // The distribution
    type Dist: Distribution<Tensor = Tensor>;

    /// Retriesve the underlying distribution
    fn distribution(&self) -> &Self::Dist;

    /// Instruments learnging with the rollout buffers collected
    fn learn(&mut self, rollouts: Vec<RolloutBuffer>) -> Result<()>;
}
