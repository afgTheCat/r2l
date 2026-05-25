use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::a2c::A2C;
use r2l_burn::{
    distributions::PolicyKind, learning_module::PolicyValueModuleKind as BurnPolicyValueModuleKind,
};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{buffers::TrajectoryBatch, on_policy::algorithm::Agent};

use crate::hooks::a2c::DefaultA2CHook;

/// A2C agent specialized to the Burn backend.
///
/// This is the concrete agent type produced by
/// [`A2CBurnAgentBuilder`](crate::A2CBurnAgentBuilder) and
/// [`A2CBurnAlgorithmBuilder`](crate::A2CBurnAlgorithmBuilder). It wraps the
/// core [`A2C`](r2l_agents::on_policy_algorithms::a2c::A2C) implementation
/// with Burn learning modules and the default A2C training hook.
///
/// Use this type when you want an [`Agent`](r2l_core::on_policy::algorithm::Agent)
/// backed by Burn instead of the default Candle backend.
pub struct A2CBurnAgent<B: AutodiffBackend>(
    pub A2C<BurnPolicyValueModuleKind<B>, DefaultA2CHook<BurnPolicyValueModuleKind<B>>>,
);

impl<B: AutodiffBackend> Agent for A2CBurnAgent<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <PolicyKind<B> as AutodiffModule<B>>::InnerModule;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}

/// A2C agent specialized to the Candle backend.
///
/// This is the default concrete A2C agent type used by
/// [`A2CAgentBuilder`](crate::A2CAgentBuilder),
/// [`A2CCandleAgentBuilder`](crate::A2CCandleAgentBuilder), and
/// [`A2CAlgorithmBuilder`](crate::A2CAlgorithmBuilder). It wraps the core
/// [`A2C`](r2l_agents::on_policy_algorithms::a2c::A2C) implementation with
/// Candle learning modules and the default A2C training hook.
///
/// Use this type when you want an [`Agent`](r2l_core::on_policy::algorithm::Agent)
/// on the default Candle backend, optionally selecting a device through
/// [`with_candle`](crate::A2CAlgorithmBuilder::with_candle).
pub struct A2CCandleAgent(
    pub A2C<CandlePolicyValueModule, DefaultA2CHook<CandlePolicyValueModule>>,
);

impl Agent for A2CCandleAgent {
    type Tensor = candle_core::Tensor;
    type Actor = CandlePolicyKind;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}
