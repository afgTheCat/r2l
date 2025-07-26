use super::{Policy, PolicyWithValueFunction};
use crate::{
    distributions::DistributionKind, policies::OptimizerWithMaxGrad,
    thread_safe_sequential::ThreadSafeSequential,
};
use candle_core::{Result, Tensor};
use candle_nn::{Module, Optimizer};

#[derive(Debug)]
pub struct DecoupledActorCritic {
    pub distribution: DistributionKind,
    pub value_net: ThreadSafeSequential,
    pub policy_optimizer_with_grad: OptimizerWithMaxGrad,
    pub value_optimizer_with_grad: OptimizerWithMaxGrad,
}

impl DecoupledActorCritic {
    pub fn policy_learning_rate(&self) -> f64 {
        self.policy_optimizer_with_grad.optimizer.learning_rate()
    }
}

impl Policy for DecoupledActorCritic {
    type Dist = DistributionKind;

    fn distribution(&self) -> &Self::Dist {
        &self.distribution
    }

    fn update(&mut self, policy_loss: &Tensor, value_loss: &Tensor) -> Result<()> {
        self.policy_optimizer_with_grad.backward_step(policy_loss)?;
        self.value_optimizer_with_grad.backward_step(value_loss)?;
        Ok(())
    }
}

impl PolicyWithValueFunction for DecoupledActorCritic {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        self.value_net.forward(observation)?.squeeze(1)
    }
}
