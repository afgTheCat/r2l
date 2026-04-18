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

pub struct BurnPPO<B: AutodiffBackend>(
    pub PPO<BurnPolicyValueModuleKind<B>, DefaultPPOHook<BurnPolicyValueModuleKind<B>>>,
);

impl<B: AutodiffBackend> Agent for BurnPPO<B> {
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

pub struct CandlePPO(pub PPO<CandlePolicyValueModule, DefaultPPOHook<CandlePolicyValueModule>>);

impl Agent for CandlePPO {
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
