// use super::policies::Policy;
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::Result;

pub trait Agent2 {
    // The distribution
    type Dist: Distribution;

    /// Retriesve the underlying distribution
    fn distribution(&self) -> &Self::Dist;

    /// Instruments learnging with the rollout buffers collected
    fn learn(&mut self, rollouts: Vec<RolloutBuffer>) -> Result<()>;
}
