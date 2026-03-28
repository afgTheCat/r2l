pub mod a2c;
pub mod ppo;

use candle_core::Tensor as CandleTensor;
use r2l_candle_lm::{
    distributions::DistributionKind,
    learning_module::{
        DecoupledActorCriticLM2, ParalellActorCriticLM2, PolicyValuesLosses,
        SequentialValueFunction,
    },
};
use r2l_core::{
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
};

pub trait ModuleWithValueFunction {
    type P: Policy<Tensor = CandleTensor>;
    type L: LearningModule<Losses = PolicyValuesLosses>;
    type V: ValueFunction<Tensor = CandleTensor>;

    fn get_inference_policy(&self) -> Self::P;

    fn get_policy_ref(&self) -> &Self::P;

    fn learning_module(&mut self) -> &mut Self::L;

    fn value_func(&self) -> &Self::V;
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
    type L = L;
    type V = SequentialValueFunction;

    fn get_inference_policy(&self) -> Self::P {
        self.policy.clone()
    }

    fn get_policy_ref(&self) -> &Self::P {
        &self.policy
    }

    fn learning_module(&mut self) -> &mut Self::L {
        &mut self.learning_module
    }

    fn value_func(&self) -> &Self::V {
        &self.value_function
    }
}

pub enum ActorCriticKind {
    Decoupled(DecoupledActorCriticLM2),
    Paralell(ParalellActorCriticLM2),
}

impl ActorCriticKind {
    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Decoupled(decoupled) => decoupled.policy_learning_rate(),
            Self::Paralell(paralell) => paralell.policy_learning_rate(),
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

pub type LearningModuleKind =
    GenericLearningModuleWithValueFunction<DistributionKind, ActorCriticKind>;
