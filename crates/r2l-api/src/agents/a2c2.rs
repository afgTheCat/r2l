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
