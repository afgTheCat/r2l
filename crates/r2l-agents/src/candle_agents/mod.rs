// pub mod a2c;
// pub mod ppo;

use anyhow::Result;
use candle_core::Tensor as CandleTensor;
use r2l_candle_lm::{
    distributions::CandleDistributionKind,
    learning_module::{
        DecoupledActorCriticLM, ParalellActorCriticLM, PolicyValuesLosses, SequentialValueFunction,
    },
};
use r2l_core::{
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
};

pub trait ModuleWithValueFunction {
    type P: Policy<Tensor = CandleTensor>;
    type V: ValueFunction<Tensor = CandleTensor>;

    fn get_inference_policy(&self) -> Self::P;

    fn get_policy_ref(&self) -> &Self::P;

    fn value_func(&self) -> &Self::V;

    fn update(&mut self, losses: PolicyValuesLosses) -> Result<()>;
}

pub struct GenericLearningModuleWithValueFunction<
    P: Policy<Tensor = CandleTensor> + Clone,
    L: LearningModule<Losses = PolicyValuesLosses>,
> {
    pub policy: P,
    pub learning_module: L,
    pub value_function: SequentialValueFunction,
}

impl<P: Policy<Tensor = CandleTensor> + Clone, L: LearningModule<Losses = PolicyValuesLosses>>
    ModuleWithValueFunction for GenericLearningModuleWithValueFunction<P, L>
{
    type P = P;
    type V = SequentialValueFunction;

    fn get_inference_policy(&self) -> Self::P {
        self.policy.clone()
    }

    fn get_policy_ref(&self) -> &Self::P {
        &self.policy
    }

    fn value_func(&self) -> &Self::V {
        &self.value_function
    }

    fn update(&mut self, losses: PolicyValuesLosses) -> Result<()> {
        self.learning_module.update(losses)
    }
}

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

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        match self {
            Self::Decoupled(lm) => lm.update(losses),
            Self::Paralell(lm) => lm.update(losses),
        }
    }
}

pub type LearningModuleKind =
    GenericLearningModuleWithValueFunction<CandleDistributionKind, ActorCriticKind>;

impl LearningModuleKind {
    pub fn policy_learning_rate(&self) -> f64 {
        self.learning_module.policy_learning_rate()
    }
}
