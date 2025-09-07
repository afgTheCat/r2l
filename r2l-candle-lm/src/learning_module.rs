use crate::{
    optimizer::OptimizerWithMaxGrad,
    tensors::{PolicyLoss, ValueLoss},
    thread_safe_sequential::ThreadSafeSequential,
};
use anyhow::{Ok, Result};
use candle_core::Tensor;
use candle_nn::{Module, Optimizer};
use r2l_core::policies::{LearningModule, ValueFunction};

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

impl DecoupledActorCriticLM {
    pub fn policy_learning_rate(&self) -> f64 {
        self.policy_optimizer_with_grad.optimizer.learning_rate()
    }
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
    type Tensor = Tensor;

    fn calculate_values(&self, observations: &[Tensor]) -> Result<Tensor> {
        let observations = Tensor::stack(observations, 0)?;
        let value = self.value_net.forward(&observations)?.squeeze(1)?;
        Ok(value)
    }
}

pub struct ParalellActorCriticLM {
    pub value_net: ThreadSafeSequential,
    pub optimizer_with_grad: OptimizerWithMaxGrad,
}

impl ParalellActorCriticLM {
    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer_with_grad.optimizer.learning_rate()
    }
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
    type Tensor = Tensor;

    fn calculate_values(&self, observations: &[Tensor]) -> Result<Tensor> {
        let observations = Tensor::stack(observations, 0)?;
        let value = self.value_net.forward(&observations)?.squeeze(1)?;
        Ok(value)
    }
}

pub enum LearningModuleKind {
    Decoupled(DecoupledActorCriticLM),
    Paralell(ParalellActorCriticLM),
}

impl LearningModuleKind {
    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Decoupled(decoupled) => decoupled.policy_learning_rate(),
            Self::Paralell(paralell) => paralell.policy_learning_rate(),
        }
    }
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
    type Tensor = Tensor;

    fn calculate_values(&self, observation: &[Tensor]) -> Result<Tensor> {
        match self {
            Self::Decoupled(decoupled) => decoupled.calculate_values(observation),
            Self::Paralell(paralell) => paralell.calculate_values(observation),
        }
    }
}
