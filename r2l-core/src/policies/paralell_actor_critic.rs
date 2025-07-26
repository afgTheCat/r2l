use super::{Policy, PolicyWithValueFunction};
use crate::policies::OptimizerWithMaxGrad;
use crate::{distributions::DistributionKind, thread_safe_sequential::ThreadSafeSequential};
use candle_core::{Result, Tensor};
use candle_nn::{Module, Optimizer};

#[derive(Debug)]
pub struct ParalellActorCritic {
    distribution: DistributionKind,
    value_net: ThreadSafeSequential,
    optimizer_with_grad: OptimizerWithMaxGrad,
}

impl ParalellActorCritic {
    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer_with_grad.optimizer.learning_rate()
    }
}

impl ParalellActorCritic {
    pub fn new(
        distribution: DistributionKind,
        value_net: ThreadSafeSequential,
        optimizer_with_grad: OptimizerWithMaxGrad,
    ) -> Self {
        Self {
            distribution,
            value_net,
            optimizer_with_grad,
        }
    }
}

// TODO: just for debugging
impl Policy for ParalellActorCritic {
    type Dist = DistributionKind;

    fn distribution(&self) -> &Self::Dist {
        &self.distribution
    }

    fn update(&mut self, policy_loss: &Tensor, value_loss: &Tensor) -> Result<()> {
        let loss = policy_loss.add(value_loss)?;
        self.optimizer_with_grad.backward_step(&loss)?;
        Ok(())
    }
}

impl PolicyWithValueFunction for ParalellActorCritic {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        self.value_net.forward(observation)?.squeeze(1)
    }
}
