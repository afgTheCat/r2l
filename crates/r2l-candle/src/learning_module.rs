use anyhow::{Ok, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use r2l_core::{
    models::{LearningModule, ValueFunction},
    on_policy::{learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses},
};

use crate::{
    distributions::CandlePolicyKind,
    optimizer::OptimizerWithMaxGrad,
    thread_safe_sequential::{ThreadSafeSequential, build_sequential},
};

pub struct PolicyValueLosses {
    pub policy_loss: Tensor,
    pub value_loss: Tensor,
    pub vf_coeff: Option<f32>,
}

impl FromPolicyValueLosses<Tensor> for PolicyValueLosses {
    fn from_policy_value_losses(policy_loss: Tensor, value_loss: Tensor) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }
}

impl PolicyValueLosses {
    pub fn new(policy_loss: Tensor, value_loss: Tensor) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }

    pub fn add_entropy_loss(&mut self, entropy_loss: Tensor) -> Result<()> {
        self.policy_loss = self.policy_loss.add(&entropy_loss)?;
        Ok(())
    }

    pub fn set_vf_coeff(&mut self, vf_coeff: Option<f32>) {
        self.vf_coeff = vf_coeff;
    }
}

pub struct SplitPolicyValueOptimizer {
    policy_optimizer_with_grad: OptimizerWithMaxGrad,
    value_optimizer_with_grad: OptimizerWithMaxGrad,
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
            let value_loss = (&losses.value_loss * Tensor::full(vf_coeff, shape, device)?)?;
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
    optimizer_with_grad: OptimizerWithMaxGrad,
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
            let value_loss = (&losses.value_loss * Tensor::full(vf_coeff, shape, device)?)?;
            losses.policy_loss.add(&value_loss)?
        } else {
            losses.policy_loss.add(&losses.value_loss)?
        };
        self.optimizer_with_grad.backward_step(&loss)?;
        Ok(())
    }
}

pub(crate) struct SequentialValueFunction {
    value_net: ThreadSafeSequential,
}

impl SequentialValueFunction {
    pub fn new(
        input_dim: usize,
        layers: &[usize],
        vb: &VarBuilder,
        prefix: &str,
    ) -> candle_core::Result<Self> {
        let value_net = build_sequential(input_dim, layers, vb, prefix)?;
        candle_core::Result::Ok(Self { value_net })
    }
}

// TODO: maybe value function could be a subtrait on LearningModule?
impl ValueFunction for SequentialValueFunction {
    type Tensor = Tensor;

    fn values(&self, observations: &[Tensor]) -> Result<Tensor> {
        let observations = Tensor::stack(observations, 0)?;
        let value = self.value_net.forward(&observations)?.squeeze(1)?;
        Ok(value)
    }
}

pub enum PolicyValueOptimizer {
    Joint(JointPolicyValueOptimizer),
    Split(SplitPolicyValueOptimizer),
}

impl PolicyValueOptimizer {
    pub fn joint(
        vm: VarMap,
        params: ParamsAdamW,
        max_grad_norm: Option<f32>,
    ) -> candle_core::Result<Self> {
        let optimizer = AdamW::new(vm.all_vars(), params)?;
        let optimizer_with_grad = OptimizerWithMaxGrad::new(optimizer, max_grad_norm, vm);
        candle_core::Result::Ok(Self::Joint(JointPolicyValueOptimizer {
            optimizer_with_grad,
        }))
    }

    pub fn split(
        policy_vm: VarMap,
        critic_vm: VarMap,
        policy_params: ParamsAdamW,
        value_params: ParamsAdamW,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    ) -> candle_core::Result<Self> {
        let policy_optimizer = AdamW::new(policy_vm.all_vars(), policy_params.clone())?;
        let value_optimizer = AdamW::new(critic_vm.all_vars(), policy_params.clone())?;
        let policy_optimizer_with_grad =
            OptimizerWithMaxGrad::new(policy_optimizer, policy_max_grad_norm, policy_vm);
        let value_optimizer_with_grad =
            OptimizerWithMaxGrad::new(value_optimizer, value_max_grad_norm, critic_vm);
        candle_core::Result::Ok(Self::Split(SplitPolicyValueOptimizer {
            policy_optimizer_with_grad,
            value_optimizer_with_grad,
        }))
    }
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
    policy: CandlePolicyKind,
    optimizer: PolicyValueOptimizer,
    value_function: SequentialValueFunction,
    device: Device,
}

impl PolicyValueModule {
    pub fn build_joint(
        policy: CandlePolicyKind,
        value_hidden_layers: &[usize],
        policy_varmap: VarMap,
        max_grad_norm: Option<f32>,
        params: ParamsAdamW,
    ) -> Result<Self> {
        let device = policy.device();
        let policy_vb = VarBuilder::from_varmap(&policy_varmap, DType::F32, &device);
        let observation_size = policy.observation_size();
        let value_layers = &[value_hidden_layers, &[1]].concat();
        let value_function =
            SequentialValueFunction::new(observation_size, value_layers, &policy_vb, "value")?;
        let optimizer = PolicyValueOptimizer::joint(policy_varmap, params, max_grad_norm)?;
        Ok(Self {
            policy,
            optimizer,
            value_function,
            device,
        })
    }

    pub fn build_split(
        policy: CandlePolicyKind,
        value_hidden_layers: &[usize],
        policy_varmap: VarMap,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
        policy_params: ParamsAdamW,
        value_params: ParamsAdamW,
    ) -> Result<Self> {
        let device = policy.device();
        let observation_size = policy.observation_size();
        let critic_varmap = VarMap::new();
        let critic_vb = VarBuilder::from_varmap(&critic_varmap, DType::F32, &device);
        let value_layers = &[value_hidden_layers, &[1]].concat();
        let value_function =
            SequentialValueFunction::new(observation_size, value_layers, &critic_vb, "value")?;
        let optimizer = PolicyValueOptimizer::split(
            policy_varmap,
            critic_varmap,
            policy_params,
            value_params,
            policy_max_grad_norm,
            value_max_grad_norm,
        )?;
        Ok(Self {
            policy,
            optimizer,
            value_function,
            device: device.clone(),
        })
    }

    pub fn set_grad_clipping(&mut self, gradient_clipping: Option<f32>) {
        self.optimizer.set_grad_clipping(gradient_clipping);
    }

    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer.policy_learning_rate()
    }
}

impl ValueFunction for PolicyValueModule {
    type Tensor = Tensor;

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
    type LearningTensor = Tensor;
    type InferenceTensor = Tensor;
    type Policy = CandlePolicyKind;
    type InferencePolicy = CandlePolicyKind;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.policy.clone()
    }

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_slice(slice, slice.len(), &self.device).unwrap()
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone()
    }
}
