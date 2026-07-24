use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::ppo::PPO;
use r2l_burn::{
    distributions::PolicyKind, learning_module::PolicyValueModuleKind as BurnPolicyValueModuleKind,
};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{buffers::TrajectoryBatch, on_policy::algorithm::Agent};

use crate::hooks::ppo::DefaultPPOHook;

/// PPO agent specialized to the Burn backend.
pub struct PPOBurnAgent<B: AutodiffBackend>(
    pub PPO<BurnPolicyValueModuleKind<B>, DefaultPPOHook<BurnPolicyValueModuleKind<B>>>,
);

impl<B: AutodiffBackend> Agent for PPOBurnAgent<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <PolicyKind<B> as AutodiffModule<B>>::InnerModule;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.0.set_learning_rate(learning_rate);
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}

/// PPO agent specialized to the Candle backend.
pub struct PPOCandleAgent(
    pub PPO<CandlePolicyValueModule, DefaultPPOHook<CandlePolicyValueModule>>,
);

impl Agent for PPOCandleAgent {
    type Tensor = candle_core::Tensor;
    type Actor = CandlePolicyKind;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.0.set_learning_rate(learning_rate);
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}
