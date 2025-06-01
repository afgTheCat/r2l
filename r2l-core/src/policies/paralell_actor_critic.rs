use super::{Policy, PolicyWithValueFunction};
use crate::utils::clip_grad::clip_grad;
use crate::{distributions::DistributionKind, thread_safe_sequential::ThreadSafeSequential};
use candle_core::{Result, Tensor};
use candle_nn::{AdamW, Module, Optimizer, VarMap};

pub struct ParalellActorCritic {
    distribution: DistributionKind,
    value_net: ThreadSafeSequential,
    optimizer: AdamW,
    max_grad_norm: Option<f32>,
    varmap: VarMap,
}

impl ParalellActorCritic {
    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
    }
}

impl ParalellActorCritic {
    pub fn new(
        distribution: DistributionKind,
        value_net: ThreadSafeSequential,
        optimizer: AdamW,
        max_grad_norm: Option<f32>,
        varmap: VarMap,
    ) -> Self {
        Self {
            distribution,
            value_net,
            optimizer,
            max_grad_norm,
            varmap,
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
        let grads = if let Some(max_norm) = self.max_grad_norm {
            clip_grad(&loss, &self.varmap, max_norm)
        } else {
            loss.backward()
        }?;
        self.optimizer.step(&grads)
    }
}

impl PolicyWithValueFunction for ParalellActorCritic {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        self.value_net.forward(observation)?.squeeze(1)
    }
}
