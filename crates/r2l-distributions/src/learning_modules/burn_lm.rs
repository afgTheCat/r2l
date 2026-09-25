//! Burn learners for batched policies and value networks.
//!
//! Observations and actions use `[batch, features]`; values, returns and log
//! probabilities use `[batch, 1]`. Reduced losses use `[1, 1]`.
//! Gaussian policies use `Param::from_tensor(log_std)` to register trainable
//! log standard deviations alongside the mean network.
//!
//! ```
//! use burn::{backend::{Autodiff, NdArray}, optim::AdamWConfig};
//! use r2l_core::models::ActivationFunction;
//! use r2l_distributions::{Categorical, networks::mlp::Mlp};
//! use r2l_distributions::learning_modules::burn_lm::PolicyValueLearner;
//!
//! type B = Autodiff<NdArray>;
//! let policy = Categorical::new(Mlp::<B>::build(&[4, 8, 2], ActivationFunction::Tanh))?;
//! let learner = PolicyValueLearner::joint(
//!     policy, &[4, 8, 1], ActivationFunction::Tanh, &AdamWConfig::new(), 0.001,
//! );
//! # Ok::<(), r2l_core::error::Error>(())
//! ```

use burn::{
    grad_clipping::GradientClipping,
    module::{AutodiffModule, Module, ModuleDisplay, Param},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_core::{
    error::Result,
    models::{ActivationFunction, Learner},
    on_policy::losses::FromPolicyValueLosses,
};

pub use crate::networks::burn::NetworkKind;
use crate::{
    DistributionKind, Network, OnPolicyLearner, Policy, ValueFunction, networks::mlp::Mlp,
};

/// Burn distributions with optimizer-managed Gaussian log standard deviations.
pub type BurnDistributionKind<B> = DistributionKind<NetworkKind<B>, Param<Tensor<B, 2>>>;

// Constraints needed for the policy to work with Adam optimization and decoupled weight decay.
/// Trait alias-like bound for Burn policies used by on-policy learners.
///
/// This captures the combination of Burn autodiff support and batched
/// [`Policy`] behavior required by the Burn learner implementations.
pub trait BurnPolicy<B: AutodiffBackend>:
    AutodiffModule<B, InnerModule: ModuleDisplay + Policy<Tensor = Tensor<B::InnerBackend, 2>>>
    + ModuleDisplay
    + Policy<Tensor = Tensor<B, 2>>
{
}

impl<B: AutodiffBackend, M> BurnPolicy<B> for M where
    M: AutodiffModule<B, InnerModule: ModuleDisplay + Policy<Tensor = Tensor<B::InnerBackend, 2>>>
        + ModuleDisplay
        + Policy<Tensor = Tensor<B, 2>>
{
}

/// Loss container used by Burn on-policy learners.
///
/// This stores the policy loss, value loss, and an optional multiplier applied
/// to the value loss during optimization.
pub struct PolicyValueLosses<B: AutodiffBackend> {
    /// Policy loss to optimize.
    pub policy_loss: Tensor<B, 2>,
    /// Value-function loss to optimize.
    pub value_loss: Tensor<B, 2>,
    /// Optional coefficient applied to `value_loss`.
    pub vf_coeff: Option<f32>,
}

impl<B: AutodiffBackend> FromPolicyValueLosses<Tensor<B, 2>> for PolicyValueLosses<B> {
    fn from_policy_value_losses(policy_loss: Tensor<B, 2>, value_loss: Tensor<B, 2>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }
}

impl<B: AutodiffBackend> PolicyValueLosses<B> {
    /// Creates a loss container from policy and value losses.
    pub fn new(policy_loss: Tensor<B, 2>, value_loss: Tensor<B, 2>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }

    /// Adds an entropy term into the policy loss.
    pub fn add_entropy_loss(&mut self, entropy_loss: Tensor<B, 2>) {
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
    value_net: NetworkKind<B>,
}

impl<B: Backend, M: Module<B>> JointActorModel<B, M> {
    /// Creates a joint model from a policy and value network.
    ///
    /// # Panics
    /// Panics unless the value network outputs one value per observation.
    pub fn new(policy: M, value_net: impl Into<NetworkKind<B>>) -> Self {
        let value_net = value_net.into();
        assert_eq!(
            value_net.output_shape().dims(),
            [1],
            "value network must output one value"
        );
        Self { policy, value_net }
    }
}

/// Burn on-policy learner with one shared optimizer configuration.
pub struct JointPolicyValueLearner<B: AutodiffBackend, M: BurnPolicy<B>> {
    lr: f64,
    model: JointActorModel<B, M>,
    // NOTE: the optimizer needs to be optimizing both the policy and the value net at the same time
    optimizer: OptimizerAdaptor<AdamW, JointActorModel<B, M>, B>,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> JointPolicyValueLearner<B, M> {
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

    /// Sets the learning rate for the shared optimizer.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.lr = learning_rate;
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> Learner for JointPolicyValueLearner<B, M> {
    type Losses = PolicyValueLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
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

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for JointPolicyValueLearner<B, M> {
    type Tensor = Tensor<B, 2>;

    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        self.model.value_net.forward(observations)
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearner for JointPolicyValueLearner<B, D> {
    type LearningTensor = Tensor<B, 2>;
    type InferenceTensor = Tensor<B::InnerBackend, 2>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.model.policy.valid()
    }

    fn policy(&self) -> &Self::Policy {
        &self.model.policy
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.set_learning_rate(learning_rate);
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::LearningTensor> {
        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(slice.to_vec(), [slice.len(), 1]),
            &self.model.value_net.devices()[0],
        ))
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_inner(t.clone())
    }
}

/// Burn on-policy learner with separate policy and value optimizers.
pub struct SplitPolicyValueLearner<B: AutodiffBackend, M: BurnPolicy<B>> {
    policy: M,
    value_net: NetworkKind<B>,
    policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
    policy_lr: f64,
    value_optimizer: OptimizerAdaptor<AdamW, NetworkKind<B>, B>,
    value_lr: f64,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> SplitPolicyValueLearner<B, M> {
    fn new(
        policy: M,
        value_net: NetworkKind<B>,
        policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
        policy_lr: f64,
        value_optimizer: OptimizerAdaptor<AdamW, NetworkKind<B>, B>,
        value_lr: f64,
    ) -> Self {
        assert_eq!(
            value_net.output_shape().dims(),
            [1],
            "value network must output one value"
        );
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

    /// Sets the learning rate for both policy and value optimizers.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.policy_lr = learning_rate;
        self.value_lr = learning_rate;
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> Learner for SplitPolicyValueLearner<B, M> {
    type Losses = PolicyValueLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
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

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for SplitPolicyValueLearner<B, M> {
    type Tensor = Tensor<B, 2>;

    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        self.value_net.forward(observations)
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearner for SplitPolicyValueLearner<B, D> {
    type LearningTensor = Tensor<B, 2>;
    type InferenceTensor = Tensor<B::InnerBackend, 2>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.policy.valid()
    }

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.set_learning_rate(learning_rate);
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::LearningTensor> {
        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(slice.to_vec(), [slice.len(), 1]),
            &self.value_net.devices()[0],
        ))
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_inner(t.clone())
    }
}

/// Erased Burn policy/value module covering joint and split optimizer layouts.
pub enum PolicyValueLearner<B: AutodiffBackend, D: BurnPolicy<B> = BurnDistributionKind<B>> {
    /// Policy/value module with one shared optimizer configuration.
    Joint(JointPolicyValueLearner<B, D>),
    /// Policy/value module with separate policy and value optimizers.
    Split(SplitPolicyValueLearner<B, D>),
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PolicyValueLearner<B, D> {
    /// Builds a policy/value module with a shared optimizer configuration.
    ///
    /// # Panics
    /// Panics for invalid layer widths or a value output width other than one.
    pub fn joint(
        policy: D,
        value_layers: &[usize],
        activation: ActivationFunction,
        optimizer_config: &AdamWConfig,
        lr: f64,
    ) -> Self {
        Self::joint_with_network(
            policy,
            NetworkKind::Mlp(Mlp::build(value_layers, activation)),
            optimizer_config,
            lr,
        )
    }

    /// Builds a joint learner with an independently constructed value network.
    ///
    /// # Panics
    /// Panics unless the value network outputs one value per observation.
    pub fn joint_with_network(
        policy: D,
        value_net: NetworkKind<B>,
        optimizer_config: &AdamWConfig,
        lr: f64,
    ) -> Self {
        let model = JointActorModel::new(policy, value_net);
        let model = JointPolicyValueLearner::new(model, optimizer_config.init(), lr);
        Self::Joint(model)
    }

    /// Builds a policy/value module with separate policy and value optimizers.
    ///
    /// # Panics
    /// Panics for invalid layer widths or a value output width other than one.
    pub fn split(
        policy: D,
        value_layers: &[usize],
        activation: ActivationFunction,
        policy_optimizer_config: &AdamWConfig,
        policy_lr: f64,
        value_optimizer_config: &AdamWConfig,
        value_lr: f64,
    ) -> Self {
        Self::split_with_network(
            policy,
            NetworkKind::Mlp(Mlp::build(value_layers, activation)),
            policy_optimizer_config,
            policy_lr,
            value_optimizer_config,
            value_lr,
        )
    }

    /// Builds a split learner with an independently constructed value network.
    ///
    /// # Panics
    /// Panics unless the value network outputs one value per observation.
    pub fn split_with_network(
        policy: D,
        value_net: NetworkKind<B>,
        policy_optimizer_config: &AdamWConfig,
        policy_lr: f64,
        value_optimizer_config: &AdamWConfig,
        value_lr: f64,
    ) -> Self {
        let model = SplitPolicyValueLearner::new(
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

impl<B: AutodiffBackend, D: BurnPolicy<B>> PolicyValueLearner<B, D> {
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

    /// Sets the learning rate on the contained optimizer state.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        match self {
            Self::Joint(lm) => lm.set_learning_rate(learning_rate),
            Self::Split(lm) => lm.set_learning_rate(learning_rate),
        }
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> Learner for PolicyValueLearner<B, M> {
    type Losses = PolicyValueLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        match self {
            Self::Joint(lm) => lm.update(losses),
            Self::Split(lm) => lm.update(losses),
        }
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for PolicyValueLearner<B, M> {
    type Tensor = Tensor<B, 2>;

    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Joint(lm) => lm.values(observations),
            Self::Split(lm) => lm.values(observations),
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearner for PolicyValueLearner<B, D> {
    type LearningTensor = Tensor<B, 2>;
    type InferenceTensor = Tensor<B::InnerBackend, 2>;
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

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.set_learning_rate(learning_rate);
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::LearningTensor> {
        match self {
            Self::Joint(lm) => lm.tensor_from_slice(slice),
            Self::Split(lm) => lm.tensor_from_slice(slice),
        }
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_inner(t.clone())
    }
}
