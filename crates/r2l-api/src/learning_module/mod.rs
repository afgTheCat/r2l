use r2l_candle_lm::{
    distributions::CandleDistributionKind,
    learning_module::{
        DecoupledActorCriticLM, ParalellActorCriticLM, PolicyValuesLosses, SequentialValueFunction,
    },
};
use r2l_core::policies::LearningModule;

pub enum ActorCriticKind {
    Decoupled(DecoupledActorCriticLM),
    Paralell(ParalellActorCriticLM),
}

impl ActorCriticKind {
    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Decoupled(decoupled) => decoupled.policy_learning_rate(),
            Self::Paralell(paralell) => paralell.policy_learning_rate(),
        }
    }

    pub fn set_grad_clipping(&mut self, max_grad_norm: Option<f32>) {
        match self {
            Self::Decoupled(decoupled) => decoupled.set_policy_grad_clip(max_grad_norm),
            Self::Paralell(paralell) => paralell.set_grad_clip(max_grad_norm),
        }
    }
}

impl LearningModule for ActorCriticKind {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        match self {
            Self::Decoupled(lm) => lm.update(losses),
            Self::Paralell(lm) => lm.update(losses),
        }
    }
}
