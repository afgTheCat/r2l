use anyhow::{Ok, Result};
use candle_core::{Device, Tensor as CandleTensor};
use candle_nn::{Module, Optimizer};
use r2l_core::{
    models::{LearningModule, ValueFunction},
    on_policy::{learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses},
};

use crate::{
    distributions::CandlePolicyKind, optimizer::OptimizerWithMaxGrad,
    thread_safe_sequential::ThreadSafeSequential,
};

pub struct PolicyValueLosses {
    pub policy_loss: CandleTensor,
    pub value_loss: CandleTensor,
    pub vf_coeff: Option<f32>,
}

impl FromPolicyValueLosses<candle_core::Tensor> for PolicyValueLosses {
    fn from_policy_value_losses(
        policy_loss: candle_core::Tensor,
        value_loss: candle_core::Tensor,
    ) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }
}

impl PolicyValueLosses {
    pub fn new(policy_loss: CandleTensor, value_loss: CandleTensor) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }

    pub fn add_entropy_loss(&mut self, entropy_loss: CandleTensor) -> Result<()> {
        self.policy_loss = self.policy_loss.add(&entropy_loss)?;
        Ok(())
    }

    pub fn set_vf_coeff(&mut self, vf_coeff: Option<f32>) {
        self.vf_coeff = vf_coeff;
    }
}

pub struct SplitPolicyValueOptimizer {
    pub policy_optimizer_with_grad: OptimizerWithMaxGrad,
    pub value_optimizer_with_grad: OptimizerWithMaxGrad,
}

impl SplitPolicyValueOptimizer {
    pub fn policy_learning_rate(&self) -> f64 {
        self.policy_optimizer_with_grad.optimizer.learning_rate()
    }

    pub fn set_policy_grad_clip(&mut self, max_grad_norm: Option<f32>) {
        self.policy_optimizer_with_grad
            .set_max_grad_norm(max_grad_norm);
    }

    pub fn set_value_grad_clip(&mut self, max_grad_norm: Option<f32>) {
        self.value_optimizer_with_grad
            .set_max_grad_norm(max_grad_norm);
    }
}

/// The policy and the value function has different optimizers
impl LearningModule for SplitPolicyValueOptimizer {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        self.policy_optimizer_with_grad
            .backward_step(&losses.policy_loss)?;
        if let Some(vf_coeff) = losses.vf_coeff {
            let device = losses.value_loss.device();
            let shape = losses.value_loss.shape();
            let value_loss = (&losses.value_loss * CandleTensor::full(vf_coeff, shape, device)?)?;
            self.value_optimizer_with_grad.backward_step(&value_loss)?;
        } else {
            self.value_optimizer_with_grad
                .backward_step(&losses.value_loss)?;
        };
        Ok(())
    }
}

/// The policy and the value fuction has the same optimizer
/// TODO: value_net does not need to be here
pub struct JointPolicyValueOptimizer {
    pub optimizer_with_grad: OptimizerWithMaxGrad,
}

impl JointPolicyValueOptimizer {
    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer_with_grad.optimizer.learning_rate()
    }

    pub fn set_grad_clip(&mut self, max_grad_norm: Option<f32>) {
        self.optimizer_with_grad.set_max_grad_norm(max_grad_norm);
    }
}

impl LearningModule for JointPolicyValueOptimizer {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        let loss = if let Some(vf_coeff) = losses.vf_coeff {
            let device = losses.value_loss.device();
            let shape = losses.value_loss.shape();
            let value_loss = (&losses.value_loss * CandleTensor::full(vf_coeff, shape, device)?)?;
            losses.policy_loss.add(&value_loss)?
        } else {
            losses.policy_loss.add(&losses.value_loss)?
        };
        self.optimizer_with_grad.backward_step(&loss)?;
        Ok(())
    }
}

pub struct SequentialValueFunction {
    pub value_net: ThreadSafeSequential,
}

// TODO: maybe value function could be a subtrait on LearningModule?
impl ValueFunction for SequentialValueFunction {
    type Tensor = CandleTensor;

    fn values(&self, observations: &[CandleTensor]) -> Result<CandleTensor> {
        let observations = CandleTensor::stack(observations, 0)?;
        let value = self.value_net.forward(&observations)?.squeeze(1)?;
        Ok(value)
    }
}

pub enum PolicyValueOptimizer {
    Joint(JointPolicyValueOptimizer),
    Split(SplitPolicyValueOptimizer),
}

impl PolicyValueOptimizer {
    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Joint(joint) => joint.policy_learning_rate(),
            Self::Split(split) => split.policy_learning_rate(),
        }
    }

    pub fn set_grad_clipping(&mut self, max_grad_norm: Option<f32>) {
        match self {
            Self::Joint(joint) => joint.set_grad_clip(max_grad_norm),
            Self::Split(split) => split.set_policy_grad_clip(max_grad_norm),
        }
    }
}

impl LearningModule for PolicyValueOptimizer {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        match self {
            Self::Joint(lm) => lm.update(losses),
            Self::Split(lm) => lm.update(losses),
        }
    }
}

pub struct PolicyValueModule {
    pub policy: CandlePolicyKind,
    pub optimizer: PolicyValueOptimizer,
    pub value_function: SequentialValueFunction,
    pub device: Device,
}

impl PolicyValueModule {
    pub fn set_grad_clipping(&mut self, gradient_clipping: Option<f32>) {
        self.optimizer.set_grad_clipping(gradient_clipping);
    }

    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer.policy_learning_rate()
    }
}

impl ValueFunction for PolicyValueModule {
    type Tensor = CandleTensor;

    fn values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        self.value_function.values(observations)
    }
}

impl LearningModule for PolicyValueModule {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        self.optimizer.update(losses)
    }
}

impl OnPolicyLearningModule for PolicyValueModule {
    type LearningTensor = CandleTensor;
    type InferenceTensor = CandleTensor;
    type Policy = CandlePolicyKind;
    type InferencePolicy = CandlePolicyKind;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.policy.clone()
    }

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        CandleTensor::from_slice(slice, slice.len(), &self.device).unwrap()
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone()
    }
}
