pub mod decoupled_actor_critic;
pub mod learning_modules;
pub mod paralell_actor_critic;

use std::fmt::Debug;

use crate::distributions::{Distribution, DistributionKind};
use candle_core::{Result, Tensor, backprop::GradStore};
use candle_nn::{AdamW, Optimizer, VarMap};
use decoupled_actor_critic::DecoupledActorCritic;
use paralell_actor_critic::ParalellActorCritic;

fn clip_grad(t: &Tensor, varmap: &VarMap, max_norm: f32) -> Result<GradStore> {
    let mut total_norm_squared = 0.0f32;
    let mut grad_store = t.backward()?;
    let mut var_ids = vec![];
    let all_vars = varmap.all_vars();
    for var in all_vars.iter() {
        let id = var.id();
        if let Some(grad) = grad_store.get_id(id) {
            var_ids.push(id);
            let grad_norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            total_norm_squared += grad_norm_sq;
        }
    }
    let total_norm = total_norm_squared.sqrt();
    if total_norm > max_norm {
        let clip_coef = (max_norm) / (total_norm + 1e-6);
        for var_id in var_ids {
            let var = all_vars.iter().find(|t| t.id() == var_id).unwrap();
            let old_grad = grad_store.get_id(var_id).unwrap();
            let clip_coef = Tensor::full(clip_coef, old_grad.shape(), old_grad.device())?;
            let new_grad = old_grad.broadcast_mul(&clip_coef)?;
            grad_store.insert(var.as_tensor(), new_grad);
        }
    }
    Ok(grad_store)
}

// pub trait Policy: Debug {
//     type Dist: Distribution;
//
//     // retrieves the underlying distribution
//     fn distribution(&self) -> &Self::Dist;
//
//     // updates the policy according to the underlying thing
//     fn update(&mut self, policy_loss: &Tensor, value_loss: &Tensor) -> Result<()>;
// }

pub trait ValueFunction {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor>;
}

// pub trait PolicyWithValueFunction: Policy {
//     // some just needs value function
//     fn calculate_values(&self, observation: &Tensor) -> Result<Tensor>;
// }

pub struct OptimizerWithMaxGrad {
    pub optimizer: AdamW,
    pub max_grad_norm: Option<f32>,
    pub varmap: VarMap,
}

impl Debug for OptimizerWithMaxGrad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimizerWithMaxGrad")
            .field("optimizer", &self.optimizer)
            .field("max_grad_norm", &self.max_grad_norm)
            .finish()
    }
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
            clip_grad(loss, &self.varmap, max_norm)?
        } else {
            loss.backward()?
        };
        self.optimizer.step(&grads)?;
        Ok(())
    }
}

#[derive(Debug)]
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

// // NOTE: it would be better to use enum dispatch here, but it's too much for the enum_dispatch
// impl Policy for PolicyKind {
//     type Dist = DistributionKind;
//
//     fn distribution(&self) -> &Self::Dist {
//         match self {
//             Self::Decoupled(policy) => policy.distribution(),
//             Self::Paralell(policy) => policy.distribution(),
//         }
//     }
//
//     fn update(&mut self, policy_loss: &Tensor, value_loss: &Tensor) -> Result<()> {
//         match self {
//             Self::Decoupled(policy) => policy.update(policy_loss, value_loss),
//             Self::Paralell(policy) => policy.update(policy_loss, value_loss),
//         }
//     }
// }
//
// impl PolicyWithValueFunction for PolicyKind {
//     fn calculate_values(&self, observation: &Tensor) -> Result<Tensor> {
//         match self {
//             Self::Decoupled(policy) => policy.calculate_values(observation),
//             Self::Paralell(policy) => policy.calculate_values(observation),
//         }
//     }
// }
