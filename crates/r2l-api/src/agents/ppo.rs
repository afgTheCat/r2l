use crate::hooks::ppo::DefaultPPOHook;
use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::ppo::PPO;
use r2l_burn::{distributions::DistributionKind, learning_module::BurnActorCriticLMKind};
use r2l_candle::{distributions::CandleDistributionKind, learning_module::R2lCandleLearningModule};
use r2l_core::{agents::Agent, buffers::TrajectoryContainer};

pub struct BurnPPO<B: AutodiffBackend>(
    pub  PPO<
        BurnActorCriticLMKind<B, DistributionKind<B>>,
        DefaultPPOHook<BurnActorCriticLMKind<B, DistributionKind<B>>>,
    >,
);

impl<B: AutodiffBackend> Agent for BurnPPO<B> {
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

pub struct CandlePPO(pub PPO<R2lCandleLearningModule, DefaultPPOHook<R2lCandleLearningModule>>);

impl Agent for CandlePPO {
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
