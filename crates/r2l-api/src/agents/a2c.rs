use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::a2c::A2C;
use r2l_burn::{distributions::DistributionKind, learning_module::BurnActorCriticLMKind};
use r2l_candle::{distributions::CandleDistributionKind, learning_module::R2lCandleLearningModule};
use r2l_core::{agents::Agent, buffers::TrajectoryContainer};

use crate::hooks::a2c::DefaultA2CHook;

pub struct BurnA2C<B: AutodiffBackend>(
    pub  A2C<
        BurnActorCriticLMKind<B, DistributionKind<B>>,
        DefaultA2CHook<BurnActorCriticLMKind<B, DistributionKind<B>>>,
    >,
);

impl<B: AutodiffBackend> Agent for BurnA2C<B> {
    type Tensor = burn::Tensor<B::InnerBackend, 1>;
    type Actor = <DistributionKind<B> as AutodiffModule<B>>::InnerModule;

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

pub struct CandleA2C(pub A2C<R2lCandleLearningModule, DefaultA2CHook<R2lCandleLearningModule>>);

impl Agent for CandleA2C {
    type Tensor = candle_core::Tensor;
    type Actor = CandleDistributionKind;

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
