pub mod decoupled_actor_critic;
pub mod paralell_actor_critic;

use crate::{
    distributions::{Distribution, DistributionKind},
    utils::clip_grad,
};
use candle_core::{Result, Tensor};
use candle_nn::{AdamW, Optimizer, VarMap};
use decoupled_actor_critic::DecoupledActorCritic;
use paralell_actor_critic::ParalellActorCritic;

pub trait Policy {
    type Dist: Distribution;

    // retrieves the underlying distribution
    fn distribution(&self) -> &Self::Dist;

    // updates the policy according to the underlying thing
    fn update(&mut self, policy_loss: &Tensor, value_loss: &Tensor) -> Result<()>;
}

pub trait PolicyWithValueFunction: Policy {
    // some just needs value function
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor>;
}

pub struct OptimizerWithMaxGrad {
    pub optimizer: AdamW,
    pub max_grad_norm: Option<f32>,
    pub varmap: VarMap,
}

impl OptimizerWithMaxGrad {
    pub fn new(optimizer: AdamW, max_grad_norm: Option<f32>, varmap: VarMap) -> Self {
        Self {
            optimizer,
            max_grad_norm,
            varmap,
        }
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = if let Some(max_norm) = self.max_grad_norm {
            clip_grad::clip_grad(loss, &self.varmap, max_norm)?
        } else {
            loss.backward()?
        };
        self.optimizer.step(&grads)?;
        Ok(())
    }
}

pub enum PolicyKind {
    Decoupled(DecoupledActorCritic),
    Paralell(ParalellActorCritic),
}

impl PolicyKind {
    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Decoupled(p) => p.policy_learning_rate(),
            Self::Paralell(p) => p.policy_learning_rate(),
        }
    }
}

// NOTE: it would be better to use enum dispatch here, but it's too much for the enum_dispatch
impl Policy for PolicyKind {
    type Dist = DistributionKind;

    fn distribution(&self) -> &Self::Dist {
        match self {
            Self::Decoupled(policy) => policy.distribution(),
            Self::Paralell(policy) => policy.distribution(),
        }
    }

    fn update(&mut self, policy_loss: &Tensor, value_loss: &Tensor) -> Result<()> {
        match self {
            Self::Decoupled(policy) => policy.update(policy_loss, value_loss),
            Self::Paralell(policy) => policy.update(policy_loss, value_loss),
        }
    }
}

impl PolicyWithValueFunction for PolicyKind {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
        match self {
            Self::Decoupled(policy) => policy.calculate_values(observation),
            Self::Paralell(policy) => policy.calculate_values(observation),
        }
    }
}
