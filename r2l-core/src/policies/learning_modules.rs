use candle_core::{Result, Tensor};
use candle_nn::Module;

use crate::{
    policies::{OptimizerWithMaxGrad, ValueFunction},
    tensors::{PolicyLoss, ValueLoss},
    thread_safe_sequential::ThreadSafeSequential,
};

// convinience trait
pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

// I guess cloning is fine here, hope it does
pub struct PolicyValuesLosses {
    pub policy_loss: PolicyLoss,
    pub value_loss: ValueLoss,
}

pub struct DecoupledActorCriticLM {
    pub value_net: ThreadSafeSequential,
    pub policy_optimizer_with_grad: OptimizerWithMaxGrad,
    pub value_optimizer_with_grad: OptimizerWithMaxGrad,
}

impl LearningModule for DecoupledActorCriticLM {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        self.policy_optimizer_with_grad
            .backward_step(&losses.policy_loss)?;
        self.value_optimizer_with_grad
            .backward_step(&losses.value_loss)?;
        Ok(())
    }
}

// TODO: maybe value function could be a subtrait on LearningModule?
impl ValueFunction for DecoupledActorCriticLM {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        self.value_net.forward(observation)?.squeeze(1)
    }
}

pub struct ParalellActorCriticLM {
    pub value_net: ThreadSafeSequential,
    pub optimizer_with_grad: OptimizerWithMaxGrad,
}

impl LearningModule for ParalellActorCriticLM {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        let loss = losses.policy_loss.add(&losses.value_loss)?;
        self.optimizer_with_grad.backward_step(&loss)?;
        Ok(())
    }
}

impl ValueFunction for ParalellActorCriticLM {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        self.value_net.forward(observation)?.squeeze(1)
    }
}

pub enum LearningModuleKind {
    Decoupled(DecoupledActorCriticLM),
    Paralell(ParalellActorCriticLM),
}

impl LearningModule for LearningModuleKind {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        match self {
            Self::Decoupled(decoupled) => decoupled.update(losses),
            Self::Paralell(paralell) => paralell.update(losses),
        }
    }
}

impl ValueFunction for LearningModuleKind {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        match self {
            Self::Decoupled(decoupled) => decoupled.calculate_values(observation),
            Self::Paralell(paralell) => paralell.calculate_values(observation),
        }
    }
}
