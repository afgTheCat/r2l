use crate::{
    optimizer::OptimizerWithMaxGrad,
    tensors::{PolicyLoss, ValueLoss},
    thread_safe_sequential::ThreadSafeSequential,
};
use anyhow::{Ok, Result};
use candle_core::Tensor;
use candle_nn::{Module, Optimizer};
use r2l_core::policies::{LearningModule, ValueFunction};

pub struct PolicyValuesLosses {
    pub policy_loss: PolicyLoss,
    pub value_loss: ValueLoss,
}

pub struct DecoupledActorCriticLM2 {
    pub policy_optimizer_with_grad: OptimizerWithMaxGrad,
    pub value_optimizer_with_grad: OptimizerWithMaxGrad,
}

impl DecoupledActorCriticLM2 {
    pub fn policy_learning_rate(&self) -> f64 {
        self.policy_optimizer_with_grad.optimizer.learning_rate()
    }
}

/// The policy and the value function has different optimizers
impl LearningModule for DecoupledActorCriticLM2 {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        self.policy_optimizer_with_grad
            .backward_step(&losses.policy_loss)?;
        self.value_optimizer_with_grad
            .backward_step(&losses.value_loss)?;
        Ok(())
    }
}

/// The policy and the value fuction has the same optimizer
/// TODO: value_net does not need to be here
pub struct ParalellActorCriticLM2 {
    pub optimizer_with_grad: OptimizerWithMaxGrad,
}

impl ParalellActorCriticLM2 {
    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer_with_grad.optimizer.learning_rate()
    }
}

impl LearningModule for ParalellActorCriticLM2 {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        let loss = losses.policy_loss.add(&losses.value_loss)?;
        self.optimizer_with_grad.backward_step(&loss)?;
        Ok(())
    }
}

pub struct SequentialValueFunction {
    pub value_net: ThreadSafeSequential,
}

// TODO: maybe value function could be a subtrait on LearningModule?
impl ValueFunction for SequentialValueFunction {
    type Tensor = Tensor;

    fn calculate_values(&self, observations: &[Tensor]) -> Result<Tensor> {
        let observations = Tensor::stack(observations, 0)?;
        let value = self.value_net.forward(&observations)?.squeeze(1)?;
        Ok(value)
    }
}
