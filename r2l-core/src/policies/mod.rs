pub mod decoupled_actor_critic;
pub mod paralell_actor_critic;

use crate::distributions::{Distribution, DistributionKind};
use candle_core::{Result, Tensor};
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
