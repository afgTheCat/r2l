use burn::{backend::NdArray, module::AutodiffModule, tensor::backend::AutodiffBackend};
use r2l_agents::on_policy_algorithms::ppo::PPO;
use r2l_burn::{distributions::DistributionKind, learning_module::BurnActorCriticLMKind};
use r2l_candle::{distributions::CandleDistributionKind, learning_module::R2lCandleLearningModule};
use r2l_core::{
    agents::Agent,
    distributions::Actor,
    sampler::buffer::{TrajectoryContainer, wrapper::BufferWrapper},
    tensor::R2lBuffer,
};

use crate::{BurnBackend, hooks::ppo::DefaultPPOHook};

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

// TODO: debatable if we need this
#[derive(Clone)]
pub enum BurnOrCandlePPOActor {
    Burn(DistributionKind<NdArray>),
    Candle(CandleDistributionKind),
}

impl Actor for BurnOrCandlePPOActor {
    type Tensor = R2lBuffer;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Burn(d) => {
                let observation = observation.into();
                let action = d.get_action(observation)?;
                Ok(action.into())
            }
            Self::Candle(d) => {
                let observation = observation.into();
                let action = d.get_action(observation)?;
                Ok(action.into())
            }
        }
    }
}

pub enum BurnOrCandlePPO {
    Burn(BurnPPO<BurnBackend>),
    Candle(CandlePPO),
}

impl Agent for BurnOrCandlePPO {
    type Tensor = R2lBuffer;
    type Actor = BurnOrCandlePPOActor;

    fn actor(&self) -> Self::Actor {
        match self {
            Self::Burn(ppo) => BurnOrCandlePPOActor::Burn(ppo.actor()),
            Self::Candle(ppo) => BurnOrCandlePPOActor::Candle(ppo.actor()),
        }
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        match self {
            Self::Burn(ppo) => {
                let buffers = buffers
                    .as_ref()
                    .iter()
                    .map(BufferWrapper::new)
                    .collect::<Vec<_>>();
                ppo.learn(&buffers)
            }
            Self::Candle(ppo) => {
                let buffers = buffers
                    .as_ref()
                    .iter()
                    .map(BufferWrapper::new)
                    .collect::<Vec<_>>();
                ppo.learn(&buffers)
            }
        }
    }

    fn shutdown(&mut self) {
        match self {
            Self::Burn(ppo) => ppo.shutdown(),
            Self::Candle(ppo) => ppo.shutdown(),
        }
    }
}
