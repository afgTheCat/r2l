use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::a2c::A2C;
use r2l_burn::{distributions::BurnPolicyKind, learning_module::PolicyValueModule};
use r2l_candle::{
    distributions::CandlePolicyKind, learning_module::PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{buffers::TrajectoryContainer, on_policy::algorithm::Agent};

use crate::hooks::a2c::DefaultA2CHook;

pub struct BurnA2C<B: AutodiffBackend>(
    pub  A2C<
        PolicyValueModule<B, BurnPolicyKind<B>>,
        DefaultA2CHook<PolicyValueModule<B, BurnPolicyKind<B>>>,
    >,
);

impl<B: AutodiffBackend> Agent for BurnA2C<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <BurnPolicyKind<B> as AutodiffModule<B>>::InnerModule;

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

pub struct CandleA2C(pub A2C<CandlePolicyValueModule, DefaultA2CHook<CandlePolicyValueModule>>);

impl Agent for CandleA2C {
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
