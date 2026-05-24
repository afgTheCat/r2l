use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::a2c2::A2C2;
use r2l_burn::{
    distributions::PolicyKind, learning_module::PolicyValueModuleKind as BurnPolicyValueModuleKind,
};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{buffers::gen_buffer::TrajectoryBatchT, on_policy::algorithm2::Agent2};

use crate::hooks::a2c2::DefaultA2CHook2;

/// A2C2 agent specialized to the Burn backend.
///
/// This is the concrete agent type produced by
/// [`A2C2BurnAgentBuilder`](crate::A2C2BurnAgentBuilder) and
/// [`A2C2BurnAlgorithmBuilder`](crate::A2C2BurnAlgorithmBuilder). It wraps the
/// core [`A2C2`](r2l_agents::on_policy_algorithms::a2c2::A2C2) implementation
/// with Burn learning modules and the default A2C2 training hook.
///
/// Use this type when you want an [`Agent2`](r2l_core::on_policy::algorithm2::Agent2)
/// backed by Burn instead of the default Candle backend.
pub struct A2C2BurnAgent<B: AutodiffBackend>(
    pub A2C2<BurnPolicyValueModuleKind<B>, DefaultA2CHook2<BurnPolicyValueModuleKind<B>>>,
);

impl<B: AutodiffBackend> Agent2 for A2C2BurnAgent<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <PolicyKind<B> as AutodiffModule<B>>::InnerModule;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatchT<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}

/// A2C2 agent specialized to the Candle backend.
///
/// This is the default concrete A2C2 agent type used by
/// [`A2C2AgentBuilder`](crate::A2C2AgentBuilder),
/// [`A2C2CandleAgentBuilder`](crate::A2C2CandleAgentBuilder), and
/// [`A2C2AlgorithmBuilder`](crate::A2C2AlgorithmBuilder). It wraps the core
/// [`A2C2`](r2l_agents::on_policy_algorithms::a2c2::A2C2) implementation with
/// Candle learning modules and the default A2C2 training hook.
///
/// Use this type when you want an [`Agent2`](r2l_core::on_policy::algorithm2::Agent2)
/// on the default Candle backend, optionally selecting a device through
/// [`with_candle`](crate::A2C2AlgorithmBuilder::with_candle).
pub struct A2C2CandleAgent(
    pub A2C2<CandlePolicyValueModule, DefaultA2CHook2<CandlePolicyValueModule>>,
);

impl Agent2 for A2C2CandleAgent {
    type Tensor = candle_core::Tensor;
    type Actor = CandlePolicyKind;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatchT<Self::Tensor>>(&mut self, buffers: &[BT]) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}
