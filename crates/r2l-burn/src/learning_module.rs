use burn::{
    grad_clipping::GradientClipping,
    module::{AutodiffModule, Module, ModuleDisplay},
    optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_core::{
    models::{LearningModule, Policy, ValueFunction},
    on_policy::{learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses},
};

use crate::sequential::Sequential;

// A series constraints that we need for the policy to work nicely with AdamW
pub trait BurnPolicy<B: AutodiffBackend>:
    AutodiffModule<B, InnerModule: ModuleDisplay + Policy<Tensor = Tensor<B::InnerBackend, 1>>>
    + ModuleDisplay
    + Policy<Tensor = Tensor<B, 1>>
{
}

impl<B: AutodiffBackend, M> BurnPolicy<B> for M where
    M: AutodiffModule<B, InnerModule: ModuleDisplay + Policy<Tensor = Tensor<B::InnerBackend, 1>>>
        + ModuleDisplay
        + Policy<Tensor = Tensor<B, 1>>
{
}

pub struct PolicyValueLosses<B: AutodiffBackend> {
    pub policy_loss: Tensor<B, 1>,
    pub value_loss: Tensor<B, 1>,
    pub vf_coeff: Option<f32>,
}

impl<B: AutodiffBackend> FromPolicyValueLosses<Tensor<B, 1>> for PolicyValueLosses<B> {
    fn from_policy_value_losses(policy_loss: Tensor<B, 1>, value_loss: Tensor<B, 1>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }
}

impl<B: AutodiffBackend> PolicyValueLosses<B> {
    pub fn new(policy_loss: Tensor<B, 1>, value_loss: Tensor<B, 1>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }

    pub fn add_entropy_loss(&mut self, entropy_loss: Tensor<B, 1>) {
        self.policy_loss = self.policy_loss.clone() + entropy_loss;
    }

    pub fn set_vf_coeff(&mut self, vf_coeff: Option<f32>) {
        self.vf_coeff = vf_coeff;
    }
}

// a model with a value function
#[derive(Debug, Module)]
pub struct JointActorModel<B: Backend, M: Module<B>> {
    pub policy: M,
    pub value_net: Sequential<B>,
}

impl<B: Backend, M: Module<B>> JointActorModel<B, M> {
    pub fn new(policy: M, value_net: Sequential<B>) -> Self {
        Self { policy, value_net }
    }
}

pub struct JointPolicyValueModule<B: AutodiffBackend, M: BurnPolicy<B>> {
    pub lr: f64,
    pub model: JointActorModel<B, M>,
    // NOTE: the optimizer needs to be optimizing both the policy and the value net at the same time
    pub optimizer: OptimizerAdaptor<AdamW, JointActorModel<B, M>, B>,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> JointPolicyValueModule<B, M> {
    pub fn new(
        model: JointActorModel<B, M>,
        optimizer: OptimizerAdaptor<AdamW, JointActorModel<B, M>, B>,
        lr: f64,
    ) -> Self {
        Self {
            lr,
            model,
            optimizer,
        }
    }

    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        self.optimizer = self.optimizer.clone().with_grad_clipping(grad_clipping);
    }

    pub fn policy_learning_rate(&self) -> f64 {
        self.lr
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> LearningModule for JointPolicyValueModule<B, M> {
    type Losses = PolicyValueLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        let loss = if let Some(vf_coeff) = losses.vf_coeff {
            losses.policy_loss + losses.value_loss.mul_scalar(vf_coeff)
        } else {
            losses.policy_loss + losses.value_loss
        };
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        let new_model = self.optimizer.step(self.lr, self.model.clone(), grads);
        self.model = new_model;
        Ok(())
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for JointPolicyValueModule<B, M> {
    type Tensor = Tensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let observation: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let value = self.model.value_net.forward(observation);
        Ok(value.squeeze())
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for JointPolicyValueModule<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.model.policy.valid()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.model.policy
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_data(slice, &Default::default())
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }
}

pub struct SplitPolicyValueModule<B: AutodiffBackend, M: BurnPolicy<B>> {
    pub policy: M,
    pub value_net: Sequential<B>,
    pub policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
    pub policy_lr: f64,
    pub value_net_optimizer: OptimizerAdaptor<AdamW, Sequential<B>, B>,
    pub value_lr: f64,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> SplitPolicyValueModule<B, M> {
    pub fn new(
        policy: M,
        value_net: Sequential<B>,
        policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
        policy_lr: f64,
        value_net_optimizer: OptimizerAdaptor<AdamW, Sequential<B>, B>,
        value_lr: f64,
    ) -> Self {
        Self {
            policy,
            value_net,
            policy_optimizer,
            policy_lr,
            value_net_optimizer,
            value_lr,
        }
    }

    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        self.policy_optimizer = self
            .policy_optimizer
            .clone()
            .with_grad_clipping(grad_clipping);
    }

    pub fn policy_learning_rate(&self) -> f64 {
        self.policy_lr
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> LearningModule for SplitPolicyValueModule<B, M> {
    type Losses = PolicyValueLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        let policy_grads = losses.policy_loss.backward();
        let policy_grads = GradientsParams::from_grads(policy_grads, &self.policy);
        self.policy = self
            .policy_optimizer
            .step(self.policy_lr, self.policy.clone(), policy_grads);
        let value_loss = if let Some(vf_coeff) = losses.vf_coeff {
            losses.value_loss * vf_coeff
        } else {
            losses.value_loss
        };
        let value_grads = value_loss.backward();
        let value_grads = GradientsParams::from_grads(value_grads, &self.value_net);
        self.value_net =
            self.value_net_optimizer
                .step(self.value_lr, self.value_net.clone(), value_grads);
        Ok(())
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for SplitPolicyValueModule<B, M> {
    type Tensor = Tensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let observation: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let value = self.value_net.forward(observation);
        Ok(value.squeeze())
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for SplitPolicyValueModule<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.policy.valid()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_data(slice, &Default::default())
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }
}

pub enum PolicyValueModule<B: AutodiffBackend, D: BurnPolicy<B>> {
    Joint(JointPolicyValueModule<B, D>),
    Split(SplitPolicyValueModule<B, D>),
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PolicyValueModule<B, D> {
    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        match self {
            Self::Joint(lm) => lm.set_grad_clipping(grad_clipping),
            Self::Split(lm) => lm.set_grad_clipping(grad_clipping),
        }
    }

    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Joint(lm) => lm.policy_learning_rate(),
            Self::Split(lm) => lm.policy_learning_rate(),
        }
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> LearningModule for PolicyValueModule<B, M> {
    type Losses = PolicyValueLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        match self {
            Self::Joint(lm) => lm.update(losses),
            Self::Split(lm) => lm.update(losses),
        }
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for PolicyValueModule<B, M> {
    type Tensor = Tensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Joint(lm) => lm.calculate_values(observations),
            Self::Split(lm) => lm.calculate_values(observations),
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for PolicyValueModule<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        match self {
            Self::Joint(lm) => lm.get_inference_policy(),
            Self::Split(lm) => lm.get_inference_policy(),
        }
    }

    fn get_policy(&self) -> &Self::Policy {
        match self {
            Self::Joint(lm) => lm.get_policy(),
            Self::Split(lm) => lm.get_policy(),
        }
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        match self {
            Self::Joint(lm) => lm.tensor_from_slice(slice),
            Self::Split(lm) => lm.tensor_from_slice(slice),
        }
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }
}
