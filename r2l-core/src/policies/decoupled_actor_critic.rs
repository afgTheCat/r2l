use super::{Policy, PolicyWithValueFunction};
use crate::{
    distributions::DistributionKind, thread_safe_sequential::ThreadSafeSequential,
    utils::clip_grad::clip_grad,
};
use candle_core::{Result, Tensor};
use candle_nn::{AdamW, Module, Optimizer, VarMap};

#[allow(dead_code)]
pub struct DecoupledActorCritic {
    pub distribution: DistributionKind,
    pub value_net: ThreadSafeSequential,
    pub policy_optimizer: AdamW,
    pub value_optimizer: AdamW,
    // TODO: this should probably be part of the thread safe sequential at one, point, or we could create a new abstraction
    pub policy_max_grad_norm: Option<f32>,
    pub value_max_grad_norm: Option<f32>, // TODO: not here dude
    pub policy_varmap: VarMap,
    pub value_varmap: VarMap,
}

impl DecoupledActorCritic {
    pub fn policy_learning_rate(&self) -> f64 {
        self.policy_optimizer.learning_rate()
    }
}

impl Policy for DecoupledActorCritic {
    type Dist = DistributionKind;

    fn distribution(&self) -> &Self::Dist {
        &self.distribution
    }

    fn update(&mut self, policy_loss: &Tensor, value_loss: &Tensor) -> Result<()> {
        let policy_grads = if let Some(policy_max_norm) = self.policy_max_grad_norm {
            clip_grad(policy_loss, &self.policy_varmap, policy_max_norm)
        } else {
            policy_loss.backward()
        }?;
        self.policy_optimizer.step(&policy_grads)?;
        let value_loss = if let Some(value_max_norm) = self.value_max_grad_norm {
            clip_grad(value_loss, &self.policy_varmap, value_max_norm)
        } else {
            value_loss.backward()
        }?;
        self.value_optimizer.step(&value_loss)?;
        Ok(())
    }
}

impl PolicyWithValueFunction for DecoupledActorCritic {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        self.value_net.forward(observation)?.squeeze(1)
    }
}
