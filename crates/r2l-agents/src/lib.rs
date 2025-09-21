pub mod burn_agents;
pub mod candle_agents;

use crate::candle_agents::ModuleWithValueFunction;
use candle_core::Tensor;
use r2l_candle_lm::{
    distributions::DistributionKind,
    learning_module2::PolicyValuesLosses,
    learning_module2::{DecoupledActorCriticLM2, ParalellActorCriticLM2, SequentialValueFunction},
};
use r2l_core::{distributions::Policy, policies::LearningModule};

pub struct GenericLearningModuleWithValueFunction<
    P: Policy<Tensor = Tensor> + Clone,
    L: LearningModule<Losses = PolicyValuesLosses>,
> {
    pub policy: P,
    pub learning_module: L,
    pub value_function: SequentialValueFunction,
}

impl<P: Policy<Tensor = Tensor> + Clone, L: LearningModule<Losses = PolicyValuesLosses>>
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

pub struct GenericLearningModuleWithValueFunction2 {
    pub policy: Box<dyn Policy<Tensor = Tensor>>,
    pub learning_module: Box<dyn LearningModule<Losses = PolicyValuesLosses>>,
    pub value_function: SequentialValueFunction,
}
