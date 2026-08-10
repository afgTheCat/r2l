use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::ppo::PPO;
use r2l_burn::{
    distributions::BurnPolicyKind,
    learning_module::PolicyValueLearner as BurnPolicyValueLearner,
};
use r2l_candle::{
    distributions::CandlePolicyKind,
    learning_module::PolicyValueLearner as CandlePolicyValueLearner,
};
use r2l_core::{buffers::TrajectoryBatch, on_policy::algorithm::Agent};

use crate::hooks::ppo::DefaultPPOHook;

/// PPO agent specialized to the Burn backend.
pub struct PPOBurnAgent<B: AutodiffBackend>(
    pub  PPO<
        BurnPolicyValueLearner<B>,
        DefaultPPOHook<BurnPolicyValueLearner<B>>,
    >,
);

impl<B: AutodiffBackend> Agent for PPOBurnAgent<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <BurnPolicyKind<B> as AutodiffModule<B>>::InnerModule;

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
    pub PPO<CandlePolicyValueLearner, DefaultPPOHook<CandlePolicyValueLearner>>,
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
