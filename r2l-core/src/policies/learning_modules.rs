use crate::{
    policies::{OptimizerWithMaxGrad, ValueFunction},
    tensors::{PolicyLoss, ValueLoss},
    thread_safe_sequential::ThreadSafeSequential,
};
use candle_core::{Result, Tensor};
use candle_nn::Module;

// convinience trait
pub trait LearningModule {
    type Losses;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

// I guess cloning is fine here
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

impl ValueFunction for DecoupledActorCriticLM {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        self.value_net.forward(observation)?.squeeze(1)
    }
}

pub struct ParalellActorCriticLM {
    value_net: ThreadSafeSequential,
    optimizer_with_grad: OptimizerWithMaxGrad,
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
