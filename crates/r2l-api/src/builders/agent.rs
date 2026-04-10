use crate::builders::policy_distribution::ActionSpaceType;
use candle_core::Device;
use r2l_core::agents::Agent;

pub trait AgentBuilder {
    type Agent: Agent;

    // TODO: the arguments to this funciton may not be final
    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent>;
}

#[derive(Debug, Clone)]
pub enum DynamicBackend {
    Burn,
    Candle(Device),
}

impl Default for DynamicBackend {
    fn default() -> Self {
        Self::Burn
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PPOBurnBackend;

#[derive(Debug, Clone)]
pub struct PPOCandleBackend {
    pub device: Device,
}
