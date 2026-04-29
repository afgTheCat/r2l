use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::ppo::PPO;
use r2l_burn::{
    distributions::PolicyKind, learning_module::PolicyValueModuleKind as BurnPolicyValueModuleKind,
};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{buffers::TrajectoryContainer, on_policy::algorithm::Agent};

use crate::hooks::ppo::DefaultPPOHook;

/// PPO agent specialized to the Burn backend.
///
/// This is the concrete agent type produced by
/// [`PPOBurnAgentBuilder`](crate::PPOBurnAgentBuilder) and
/// [`PPOBurnAlgorithmBuilder`](crate::PPOBurnAlgorithmBuilder). It wraps the
/// core [`PPO`](r2l_agents::on_policy_algorithms::ppo::PPO) implementation
/// with Burn learning modules and the default PPO training hook.
///
/// Use this type when you want an [`Agent`](r2l_core::on_policy::algorithm::Agent)
/// backed by Burn instead of the default Candle backend.
pub struct PPOBurnAgent<B: AutodiffBackend>(
    pub PPO<BurnPolicyValueModuleKind<B>, DefaultPPOHook<BurnPolicyValueModuleKind<B>>>,
);

impl<B: AutodiffBackend> Agent for PPOBurnAgent<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <PolicyKind<B> as AutodiffModule<B>>::InnerModule;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}

/// PPO agent specialized to the Candle backend.
///
/// This is the default concrete PPO agent type used by
/// [`PPOAgentBuilder`](crate::PPOAgentBuilder),
/// [`PPOCandleAgentBuilder`](crate::PPOCandleAgentBuilder), and
/// [`PPOAlgorithmBuilder`](crate::PPOAlgorithmBuilder). It wraps the core
/// [`PPO`](r2l_agents::on_policy_algorithms::ppo::PPO) implementation with
/// Candle learning modules and the default PPO training hook.
///
/// Use this type when you want an [`Agent`](r2l_core::on_policy::algorithm::Agent)
/// on the default Candle backend, optionally selecting a device through
/// [`with_candle`](crate::PPOAlgorithmBuilder::with_candle).
pub struct PPOCandleAgent(
    pub PPO<CandlePolicyValueModule, DefaultPPOHook<CandlePolicyValueModule>>,
);

impl Agent for PPOCandleAgent {
    type Tensor = candle_core::Tensor;
    type Actor = CandlePolicyKind;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}
