//! Burn policy/value learning modules used by on-policy algorithms.
//!
//! The central public type here is [`crate::learning_module::PolicyValueModule`],
//! which combines a Burn policy, a value function, and optimizer state into one
//! [`OnPolicyLearningModule`](r2l_core::on_policy::learning_module::OnPolicyLearningModule)
//! implementation.

use burn::{
    grad_clipping::GradientClipping,
    module::{AutodiffModule, Module, ModuleDisplay},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_core::{
    models::{ActivationFunction, Actor, LearningModule, ValueFunction},
    on_policy::{learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses},
};

use crate::{distributions::PolicyKind, sequential::Sequential};

// A series constraints that we need for the policy to work nicely with AdamW
/// Trait alias-like bound for Burn policies used by on-policy learning modules.
///
/// This captures the Burn autodiff and inference-time [`Actor`] behavior
/// required by the Burn learning-module implementations. Algorithms add their
/// own [`r2l_core::models::Policy`] or recurrent-policy bounds.
pub trait BurnPolicy<B: AutodiffBackend>:
    AutodiffModule<B, InnerModule: ModuleDisplay + Actor<Tensor = Tensor<B::InnerBackend, 1>>>
    + ModuleDisplay
    + Actor<Tensor = Tensor<B, 1>>
{
    fn lift_state(state: &<Self::InnerModule as Actor>::State) -> Self::State;
}

impl<
    B: AutodiffBackend,
    M: AutodiffModule<B, InnerModule: ModuleDisplay + Actor<Tensor = Tensor<B::InnerBackend, 1>>>
        + ModuleDisplay
        + Actor<
            Tensor = Tensor<B, 1>,
            State: BurnStateLifter<B, InferenceState = <M::InnerModule as Actor>::State>,
        >,
> BurnPolicy<B> for M
{
    fn lift_state(state: &<Self::InnerModule as Actor>::State) -> Self::State {
        <M::State as BurnStateLifter<B>>::lift_state(state)
    }
}

/// Converts rollout state from Burn's inner backend to its autodiff backend.
pub trait BurnStateLifter<B: AutodiffBackend>: Clone + Send + Sync + 'static {
    type InferenceState: Clone + Send + Sync + 'static;

    fn lift_state(state: &Self::InferenceState) -> Self;
}

impl<B: AutodiffBackend> BurnStateLifter<B> for () {
    type InferenceState = ();

    fn lift_state(_state: &Self::InferenceState) -> Self {}
}

impl<B: AutodiffBackend, const D: usize> BurnStateLifter<B> for Tensor<B, D> {
    type InferenceState = Tensor<B::InnerBackend, D>;

    fn lift_state(state: &Self::InferenceState) -> Self {
        Tensor::from_data(state.to_data(), &Default::default())
    }
}

/// Loss container used by Burn on-policy learning modules.
///
/// This stores the policy loss, value loss, and an optional multiplier applied
/// to the value loss during optimization.
pub struct PolicyValueLosses<B: AutodiffBackend> {
    /// Policy loss to optimize.
    pub policy_loss: Tensor<B, 1>,
    /// Value-function loss to optimize.
    pub value_loss: Tensor<B, 1>,
    /// Optional coefficient applied to `value_loss`.
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
    /// Creates a loss container from policy and value losses.
    pub fn new(policy_loss: Tensor<B, 1>, value_loss: Tensor<B, 1>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }

    /// Adds an entropy term into the policy loss.
    pub fn add_entropy_loss(&mut self, entropy_loss: Tensor<B, 1>) {
        self.policy_loss = self.policy_loss.clone() + entropy_loss;
    }

    /// Sets the optional value-loss coefficient used during optimization.
    pub fn set_vf_coeff(&mut self, vf_coeff: Option<f32>) {
        self.vf_coeff = vf_coeff;
    }
}

// a model with a value function
/// Combined policy/value model used by the joint Burn optimizer path.
#[derive(Debug, Module)]
pub struct JointActorModel<B: Backend, M: Module<B>> {
    policy: M,
    value_net: Sequential<B>,
}

impl<B: Backend, M: Module<B>> JointActorModel<B, M> {
    /// Creates a joint model from a policy and value network.
    pub fn new(policy: M, value_net: Sequential<B>) -> Self {
        Self { policy, value_net }
    }
}

/// Burn on-policy learning module with one shared optimizer configuration.
pub struct JointPolicyValueModule<B: AutodiffBackend, M: BurnPolicy<B>> {
    lr: f64,
    model: JointActorModel<B, M>,
    // NOTE: the optimizer needs to be optimizing both the policy and the value net at the same time
    optimizer: OptimizerAdaptor<AdamW, JointActorModel<B, M>, B>,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> JointPolicyValueModule<B, M> {
    fn new(
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

    /// Sets gradient clipping for the shared optimizer.
    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        self.optimizer = self.optimizer.clone().with_grad_clipping(grad_clipping);
    }

    /// Returns the current policy optimizer learning rate.
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

    fn values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let observation: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let value = self.model.value_net.forward(observation);
        Ok(value.squeeze())
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for JointPolicyValueModule<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type InferenceState = <D::InnerModule as Actor>::State;
    type LearningState = D::State;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.model.policy.valid()
    }

    fn policy(&self) -> &Self::Policy {
        &self.model.policy
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_data(slice, &Default::default())
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }

    fn state_lifter(state: &Self::InferenceState) -> Self::LearningState {
        D::lift_state(state)
    }
}

/// Burn on-policy learning module with separate policy and value optimizers.
pub struct SplitPolicyValueModule<B: AutodiffBackend, M: BurnPolicy<B>> {
    policy: M,
    value_net: Sequential<B>,
    policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
    policy_lr: f64,
    value_optimizer: OptimizerAdaptor<AdamW, Sequential<B>, B>,
    value_lr: f64,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> SplitPolicyValueModule<B, M> {
    fn new(
        policy: M,
        value_net: Sequential<B>,
        policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
        policy_lr: f64,
        value_optimizer: OptimizerAdaptor<AdamW, Sequential<B>, B>,
        value_lr: f64,
    ) -> Self {
        Self {
            policy,
            value_net,
            policy_optimizer,
            policy_lr,
            value_optimizer,
            value_lr,
        }
    }

    /// Sets gradient clipping for the policy optimizer.
    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        self.policy_optimizer = self
            .policy_optimizer
            .clone()
            .with_grad_clipping(grad_clipping);
    }

    /// Returns the current policy optimizer learning rate.
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
            self.value_optimizer
                .step(self.value_lr, self.value_net.clone(), value_grads);
        Ok(())
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for SplitPolicyValueModule<B, M> {
    type Tensor = Tensor<B, 1>;

    fn values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let observation: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let value = self.value_net.forward(observation);
        Ok(value.squeeze())
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for SplitPolicyValueModule<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type InferenceState = <D::InnerModule as Actor>::State;
    type LearningState = D::State;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.policy.valid()
    }

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_data(slice, &Default::default())
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }

    fn state_lifter(state: &Self::InferenceState) -> Self::LearningState {
        D::lift_state(state)
    }
}

/// Erased Burn policy/value module covering joint and split optimizer layouts.
pub enum PolicyValueModule<B: AutodiffBackend, D: BurnPolicy<B>> {
    /// Policy/value module with one shared optimizer configuration.
    Joint(JointPolicyValueModule<B, D>),
    /// Policy/value module with separate policy and value optimizers.
    Split(SplitPolicyValueModule<B, D>),
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PolicyValueModule<B, D> {
    /// Builds a policy/value module with a shared optimizer configuration.
    pub fn joint(
        policy: D,
        value_layers: &[usize],
        activation: ActivationFunction,
        optimizer_config: AdamWConfig,
        lr: f64,
    ) -> Self {
        let value_net: Sequential<B> = Sequential::build(value_layers, activation);
        let model = JointActorModel::new(policy, value_net);
        let model = JointPolicyValueModule::new(model, optimizer_config.init(), lr);
        Self::Joint(model)
    }

    /// Builds a policy/value module with separate policy and value optimizers.
    pub fn split(
        policy: D,
        value_layers: &[usize],
        activation: ActivationFunction,
        policy_optimizer_config: AdamWConfig,
        policy_lr: f64,
        value_optimizer_config: AdamWConfig,
        value_lr: f64,
    ) -> Self {
        let value_net: Sequential<B> = Sequential::build(value_layers, activation);
        let model = SplitPolicyValueModule::new(
            policy,
            value_net,
            policy_optimizer_config.init(),
            policy_lr,
            value_optimizer_config.init(),
            value_lr,
        );
        Self::Split(model)
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PolicyValueModule<B, D> {
    /// Sets policy-side gradient clipping on the contained optimizer state.
    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        match self {
            Self::Joint(lm) => lm.set_grad_clipping(grad_clipping),
            Self::Split(lm) => lm.set_grad_clipping(grad_clipping),
        }
    }

    /// Returns the current policy optimizer learning rate.
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

    fn values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Joint(lm) => lm.values(observations),
            Self::Split(lm) => lm.values(observations),
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for PolicyValueModule<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type InferenceState = <D::InnerModule as Actor>::State;
    type LearningState = D::State;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn inference_policy(&self) -> Self::InferencePolicy {
        match self {
            Self::Joint(lm) => lm.inference_policy(),
            Self::Split(lm) => lm.inference_policy(),
        }
    }

    fn policy(&self) -> &Self::Policy {
        match self {
            Self::Joint(lm) => lm.policy(),
            Self::Split(lm) => lm.policy(),
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

    fn state_lifter(state: &Self::InferenceState) -> Self::LearningState {
        D::lift_state(state)
    }
}

pub type PolicyValueModuleKind<B> = PolicyValueModule<B, PolicyKind<B>>;
