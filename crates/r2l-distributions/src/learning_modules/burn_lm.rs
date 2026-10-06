//! Burn learners for batched policies and value networks.
//!
//! Observations and actions use `[batch, features]`; values, returns and log
//! probabilities use `[batch, 1]`. Reduced losses use `[1, 1]`.
//! Gaussian policies use `Param::from_tensor(log_std)` to register trainable
//! log standard deviations alongside the mean network.
//! Learner construction enables training on both networks. Inference policies
//! use `Module::valid()`, and `OnPolicyLearner::lifter` enables autodiff on batches.
//!
//! ```
//! use burn::optim::AdamWConfig;
//! use r2l_core::models::ActivationFunction;
//! use r2l_distributions::{Categorical, networks::burn::mlp::Mlp};
//! use r2l_distributions::learning_modules::burn_lm::PolicyValueLearner;
//!
//! let policy = Categorical::new(Mlp::build(&[4, 8, 2], ActivationFunction::Tanh))?;
//! let learner = PolicyValueLearner::joint(
//!     policy, &[4, 8, 1], ActivationFunction::Tanh, &AdamWConfig::new(), 0.001,
//! );
//! # Ok::<(), r2l_core::error::Error>(())
//! ```

use std::path::PathBuf;

#[cfg(test)]
mod tests;

use burn::{
    module::{Module, ModuleDisplay, Param},
    optim::{AdamWConfig, GradientsParams, ModuleOptimizer, OptimizerRecord},
    store::ModuleRecord,
    tensor::{Bytes, DType, Tensor},
};
use r2l_core::{
    error::Result,
    models::{ActivationFunction, Learner},
    on_policy::losses::FromPolicyValueLosses,
};

pub use crate::networks::burn::NetworkKind;
use crate::{
    DistributionKind, Network, OnPolicyLearner, Policy, ValueFunction, networks::burn::mlp::Mlp,
};

/// Burn distributions with optimizer-managed Gaussian log standard deviations.
pub type BurnDistributionKind = DistributionKind<NetworkKind, Param<Tensor<2>>>;

// Constraints needed for the policy to work with Adam optimization and decoupled weight decay.
/// Trait alias-like bound for Burn policies used by on-policy learners.
///
/// This captures the combination of Burn autodiff support and batched
/// [`Policy`] behavior required by the Burn learner implementations.
pub trait BurnPolicy: Module + ModuleDisplay + Policy<Tensor = Tensor<2>> {}

impl<M> BurnPolicy for M where M: Module + ModuleDisplay + Policy<Tensor = Tensor<2>> {}

/// Loss container used by Burn on-policy learners.
///
/// This stores the policy loss, value loss, and a multiplier applied
/// to the value loss during optimization.
pub struct PolicyValueLosses {
    /// Policy loss to optimize.
    pub policy_loss: Tensor<2>,
    /// Value-function loss to optimize.
    pub value_loss: Tensor<2>,
    /// Coefficient applied to `value_loss`, defaulting to `1.0`.
    pub vf_coeff: f32,
}

impl FromPolicyValueLosses<Tensor<2>> for PolicyValueLosses {
    fn from_policy_value_losses(policy_loss: Tensor<2>, value_loss: Tensor<2>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: 1.0,
        }
    }
}

impl PolicyValueLosses {
    /// Creates a loss container from policy and value losses.
    pub fn new(policy_loss: Tensor<2>, value_loss: Tensor<2>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: 1.0,
        }
    }

    /// Adds an entropy term into the policy loss.
    pub fn add_entropy_loss(&mut self, entropy_loss: Tensor<2>) {
        self.policy_loss = self.policy_loss.clone() + entropy_loss;
    }

    /// Sets the value-loss coefficient used during optimization.
    pub fn set_vf_coeff(&mut self, vf_coeff: f32) {
        self.vf_coeff = vf_coeff;
    }
}

// a model with a value function
/// Combined policy/value model used by the joint Burn optimizer path.
#[derive(Debug, Module)]
pub struct JointActorModel<M: Module> {
    policy: M,
    value_net: NetworkKind,
}

impl<M: Module> JointActorModel<M> {
    /// Creates a joint model from a policy and value network.
    ///
    /// # Panics
    /// Panics unless the value network outputs one value per observation.
    pub fn new(policy: M, value_net: impl Into<NetworkKind>) -> Self {
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
pub struct JointPolicyValueLearner<M: BurnPolicy> {
    lr: f64,
    model: JointActorModel<M>,
    // NOTE: the optimizer needs to be optimizing both the policy and the value net at the same time
    optimizer: ModuleOptimizer,
}

/// Model parameters, optimizer state, and learning rate for a joint learner.
pub struct JointPolicyValueSnapshot {
    lr: f64,
    model: ModuleRecord,
    optimizer: OptimizerRecord,
}

impl JointPolicyValueSnapshot {
    /// Saves model and optimizer records in a Burnpack snapshot file.
    ///
    /// # Panics
    /// Panics if a record cannot be serialized or the file cannot be written.
    pub fn to_file(self, file: PathBuf) {
        PolicyValueLearnerSnapshot::save_records(
            file,
            [
                ("model", self.model.into_bytes().unwrap()),
                ("optimizer", self.optimizer.into_bytes().unwrap()),
            ],
            [("lr", self.lr)],
        );
    }
}

impl<M: BurnPolicy> JointPolicyValueLearner<M> {
    fn new(model: JointActorModel<M>, optimizer: ModuleOptimizer, lr: f64) -> Self {
        Self {
            lr,
            model: model.train(),
            optimizer,
        }
    }

    pub fn load_snapshot(self, snapshot: JointPolicyValueSnapshot) -> Self {
        let JointPolicyValueSnapshot {
            lr,
            model,
            optimizer,
        } = snapshot;
        let model = self.model.load_record(model);
        let optimizer = self.optimizer.load_record(optimizer);
        Self {
            model,
            optimizer,
            lr,
        }
    }

    /// Returns the current policy optimizer learning rate.
    pub fn policy_learning_rate(&self) -> f64 {
        self.lr
    }

    /// Sets the learning rate for the shared optimizer.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.lr = learning_rate;
    }

    pub fn to_snapshot(&self) -> JointPolicyValueSnapshot {
        let model = self.model.clone().into_record();
        let optimizer = self.optimizer.to_record();
        JointPolicyValueSnapshot {
            model,
            optimizer,
            lr: self.lr,
        }
    }
}

impl<M: BurnPolicy> Learner for JointPolicyValueLearner<M> {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        let loss = losses.policy_loss + losses.value_loss.mul_scalar(losses.vf_coeff);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        let new_model = self.optimizer.step(self.lr, self.model.clone(), grads);
        self.model = new_model;
        Ok(())
    }
}

impl<M: BurnPolicy> ValueFunction for JointPolicyValueLearner<M> {
    type Tensor = Tensor<2>;

    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        self.model.value_net.forward(observations)
    }
}

impl<D: BurnPolicy> OnPolicyLearner for JointPolicyValueLearner<D> {
    type LearningTensor = Tensor<2>;
    type InferenceTensor = Tensor<2>;
    type Policy = D;
    type InferencePolicy = D;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.model.policy.valid()
    }

    fn policy(&self) -> &Self::Policy {
        &self.model.policy
    }

    fn set_learning_rates(&mut self, policy_learning_rate: f64, _value_learning_rate: f64) {
        self.lr = policy_learning_rate;
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::LearningTensor> {
        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(slice.to_vec(), [slice.len(), 1]),
            &self.model.value_net.devices()[0],
        ))
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone().autodiff()
    }
}

/// Burn on-policy learner with separate policy and value optimizers.
pub struct SplitPolicyValueLearner<M: BurnPolicy> {
    policy: M,
    value_net: NetworkKind,
    policy_optimizer: ModuleOptimizer,
    policy_lr: f64,
    value_optimizer: ModuleOptimizer,
    value_lr: f64,
}

/// Model parameters, optimizer states, and learning rates for a split learner.
pub struct SplitPolicyValueLeranerSnapshot {
    policy: ModuleRecord,
    value_net: ModuleRecord,
    policy_optimizer: OptimizerRecord,
    policy_lr: f64,
    value_optimizer: OptimizerRecord,
    value_lr: f64,
}

impl SplitPolicyValueLeranerSnapshot {
    /// Saves model and optimizer records in a Burnpack snapshot file.
    ///
    /// # Panics
    /// Panics if a record cannot be serialized or the file cannot be written.
    pub fn to_file(self, file: PathBuf) {
        PolicyValueLearnerSnapshot::save_records(
            file,
            [
                ("policy", self.policy.into_bytes().unwrap()),
                ("value_net", self.value_net.into_bytes().unwrap()),
                (
                    "policy_optimizer",
                    self.policy_optimizer.into_bytes().unwrap(),
                ),
                (
                    "value_optimizer",
                    self.value_optimizer.into_bytes().unwrap(),
                ),
            ],
            [("policy_lr", self.policy_lr), ("value_lr", self.value_lr)],
        );
    }
}

impl<M: BurnPolicy> SplitPolicyValueLearner<M> {
    fn new(
        policy: M,
        value_net: NetworkKind,
        policy_optimizer: ModuleOptimizer,
        policy_lr: f64,
        value_optimizer: ModuleOptimizer,
        value_lr: f64,
    ) -> Self {
        assert_eq!(
            value_net.output_shape().dims(),
            [1],
            "value network must output one value"
        );
        Self {
            policy: policy.train(),
            value_net: value_net.train(),
            policy_optimizer,
            policy_lr,
            value_optimizer,
            value_lr,
        }
    }

    fn load_snapshot(self, snapshot: SplitPolicyValueLeranerSnapshot) -> Self {
        let SplitPolicyValueLeranerSnapshot {
            policy,
            value_net,
            policy_optimizer,
            policy_lr,
            value_optimizer,
            value_lr,
        } = snapshot;
        let policy = self.policy.load_record(policy);
        let value_net = self.value_net.load_record(value_net);
        let policy_optimizer = self.policy_optimizer.load_record(policy_optimizer);
        let value_optimizer = self.value_optimizer.load_record(value_optimizer);
        Self {
            policy,
            value_net,
            policy_optimizer,
            policy_lr,
            value_optimizer,
            value_lr,
        }
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

impl<M: BurnPolicy> Learner for SplitPolicyValueLearner<M> {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        let policy_grads = losses.policy_loss.backward();
        let policy_grads = GradientsParams::from_grads(policy_grads, &self.policy);
        self.policy = self
            .policy_optimizer
            .step(self.policy_lr, self.policy.clone(), policy_grads);
        let value_loss = losses.value_loss * losses.vf_coeff;
        let value_grads = value_loss.backward();
        let value_grads = GradientsParams::from_grads(value_grads, &self.value_net);
        self.value_net =
            self.value_optimizer
                .step(self.value_lr, self.value_net.clone(), value_grads);
        Ok(())
    }
}

impl<M: BurnPolicy> ValueFunction for SplitPolicyValueLearner<M> {
    type Tensor = Tensor<2>;

    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        self.value_net.forward(observations)
    }
}

impl<D: BurnPolicy> OnPolicyLearner for SplitPolicyValueLearner<D> {
    type LearningTensor = Tensor<2>;
    type InferenceTensor = Tensor<2>;
    type Policy = D;
    type InferencePolicy = D;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.policy.valid()
    }

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64) {
        self.policy_lr = policy_learning_rate;
        self.value_lr = value_learning_rate;
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::LearningTensor> {
        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(slice.to_vec(), [slice.len(), 1]),
            &self.value_net.devices()[0],
        ))
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone().autodiff()
    }
}

/// Snapshot state matching the learner's optimizer layout.
pub enum PolicyValueLearnerSnapshot {
    Joint(JointPolicyValueSnapshot),
    Split(SplitPolicyValueLeranerSnapshot),
}

impl PolicyValueLearnerSnapshot {
    // Store each Burn record as a byte tensor, with learning rates as typed scalars.
    fn save_records<const N: usize, const L: usize>(
        file: PathBuf,
        records: [(&str, Bytes); N],
        learning_rates: [(&str, f64); L],
    ) {
        let tensors = records
            .into_iter()
            .map(|(name, bytes)| {
                let shape = [bytes.len()];
                burn_pack::Tensor::new(name.into(), DType::U8, shape, None, bytes)
            })
            .collect();
        let writer = learning_rates
            .into_iter()
            .fold(burn_pack::Writer::new(tensors), |writer, (name, lr)| {
                writer.with_scalar(name, burn_pack::Scalar::Float(lr))
            });
        writer.write_to_file(file).unwrap();
    }
}

/// Erased Burn policy/value module covering joint and split optimizer layouts.
pub enum PolicyValueLearner<D: BurnPolicy = BurnDistributionKind> {
    /// Policy/value module with one shared optimizer configuration.
    Joint(JointPolicyValueLearner<D>),
    /// Policy/value module with separate policy and value optimizers.
    Split(SplitPolicyValueLearner<D>),
}

impl<D: BurnPolicy> PolicyValueLearner<D> {
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

    pub fn load_snapshot(self, snapshot: PolicyValueLearnerSnapshot) -> Self {
        match snapshot {
            PolicyValueLearnerSnapshot::Joint(joint_policy_value_snapshot) => {
                let Self::Joint(joint) = self else { panic!() };
                let joint = joint.load_snapshot(joint_policy_value_snapshot);
                Self::Joint(joint)
            }
            PolicyValueLearnerSnapshot::Split(split_policy_value_leraner_snapshot) => {
                let Self::Split(split) = self else { panic!() };
                let split = split.load_snapshot(split_policy_value_leraner_snapshot);
                Self::Split(split)
            }
        }
    }

    /// Builds a joint learner with an independently constructed value network.
    ///
    /// # Panics
    /// Panics unless the value network outputs one value per observation.
    pub fn joint_with_network(
        policy: D,
        value_net: NetworkKind,
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
        value_net: NetworkKind,
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

impl<D: BurnPolicy> PolicyValueLearner<D> {
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

impl<M: BurnPolicy> Learner for PolicyValueLearner<M> {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        match self {
            Self::Joint(lm) => lm.update(losses),
            Self::Split(lm) => lm.update(losses),
        }
    }
}

impl<M: BurnPolicy> ValueFunction for PolicyValueLearner<M> {
    type Tensor = Tensor<2>;

    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Joint(lm) => lm.values(observations),
            Self::Split(lm) => lm.values(observations),
        }
    }
}

impl<D: BurnPolicy> OnPolicyLearner for PolicyValueLearner<D> {
    type LearningTensor = Tensor<2>;
    type InferenceTensor = Tensor<2>;
    type Policy = D;
    type InferencePolicy = D;

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

    fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64) {
        match self {
            Self::Joint(lm) => lm.set_learning_rates(policy_learning_rate, value_learning_rate),
            Self::Split(lm) => lm.set_learning_rates(policy_learning_rate, value_learning_rate),
        }
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::LearningTensor> {
        match self {
            Self::Joint(lm) => lm.tensor_from_slice(slice),
            Self::Split(lm) => lm.tensor_from_slice(slice),
        }
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        t.clone().autodiff()
    }
}
