pub mod a2c;
pub mod ppo;
pub mod vpg;

use super::{distributions::Distribution, policies::Policy};
use crate::utils::rollout_buffer::RolloutBuffer;
use candle_core::Result;

pub trait Agent {
    /// Retrieves the underlying policy
    fn policy(&self) -> &impl Policy;

    /// Retriesve the underlying distribution throught the policy
    fn distribution(&self) -> &impl Distribution {
        self.policy().distribution()
    }

    /// Instruments learnging with the rollout buffers collected
    fn learn(&mut self, rollout_buffer: Vec<RolloutBuffer>) -> Result<()>;
}
