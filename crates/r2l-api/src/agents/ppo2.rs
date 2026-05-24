use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::ppo2::PPO2;
use r2l_burn::{
    distributions::PolicyKind, learning_module::PolicyValueModuleKind as BurnPolicyValueModuleKind,
};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{buffers::gen_buffer::TrajectoryBatchT, on_policy::algorithm2::Agent2};

use crate::hooks::ppo2::DefaultPPO2Hook;

/// PPO2 agent specialized to the Burn backend.
pub struct PPO2BurnAgent<B: AutodiffBackend>(
    pub PPO2<BurnPolicyValueModuleKind<B>, DefaultPPO2Hook<BurnPolicyValueModuleKind<B>>>,
);

impl<B: AutodiffBackend> Agent2 for PPO2BurnAgent<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <PolicyKind<B> as AutodiffModule<B>>::InnerModule;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatchT<Self::Tensor>>(
        &mut self,
        buffers: &[BT],
    ) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}

/// PPO2 agent specialized to the Candle backend.
pub struct PPO2CandleAgent(
    pub PPO2<CandlePolicyValueModule, DefaultPPO2Hook<CandlePolicyValueModule>>,
);

impl Agent2 for PPO2CandleAgent {
    type Tensor = candle_core::Tensor;
    type Actor = CandlePolicyKind;

    fn actor(&self) -> Self::Actor {
        self.0.actor()
    }

    fn learn<BT: TrajectoryBatchT<Self::Tensor>>(
        &mut self,
        buffers: &[BT],
    ) -> anyhow::Result<()> {
        self.0.learn(buffers)
    }

    fn shutdown(&mut self) {
        self.0.shutdown();
    }
}
