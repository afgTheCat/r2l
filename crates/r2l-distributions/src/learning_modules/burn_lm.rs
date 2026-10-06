//! Burn learners for batched policies and value networks.
//!
//! Observations and actions use `[batch, features]`; values, returns and log
//! probabilities use `[batch, 1]`. Reduced losses use `[1, 1]`.
//! Gaussian policies use `Param::from_tensor(log_std)` to register trainable
//! log standard deviations alongside the mean network.
//! Learner construction enables training on both networks. Inference policies
//! use `Module::valid()`, and `OnPolicyLearner::prepare_learning_tensor` enables autodiff on batches.
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

/// Shared policy/value losses specialized to this backend.
pub type PolicyValueLosses = r2l_core::on_policy::losses::PolicyValueLosses<Tensor<2>>;

#[derive(Debug, Module)]
struct PolicyValueModel<P: Module> {
    policy: P,
    value_net: NetworkKind,
}

enum OptimizerKind<O> {
    Joint {
        optimizer: O,
        lr: f64,
    },
    Split {
        policy: O,
        policy_lr: f64,
        value: O,
        value_lr: f64,
    },
}

/// Joint or split Burn optimizers, with a learning rate for each optimizer.
pub struct PolicyValueOptimizer {
    inner: OptimizerKind<ModuleOptimizer>,
}

impl PolicyValueOptimizer {
    /// Builds one optimizer for both policy and value parameters.
    #[must_use]
    pub fn joint(config: &AdamWConfig, lr: f64) -> Self {
        Self {
            inner: OptimizerKind::Joint {
                optimizer: config.init(),
                lr,
            },
        }
    }

    /// Builds separate optimizers for independent policy and value networks.
    #[must_use]
    pub fn split(
        policy_config: &AdamWConfig,
        policy_lr: f64,
        value_config: &AdamWConfig,
        value_lr: f64,
    ) -> Self {
        Self {
            inner: OptimizerKind::Split {
                policy: policy_config.init(),
                policy_lr,
                value: value_config.init(),
                value_lr,
            },
        }
    }

    /// Returns the current policy optimizer learning rate.
    #[must_use]
    pub fn policy_learning_rate(&self) -> f64 {
        match &self.inner {
            OptimizerKind::Joint { lr, .. } => *lr,
            OptimizerKind::Split { policy_lr, .. } => *policy_lr,
        }
    }

    /// Sets the same learning rate for both policy and value updates.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.set_learning_rates(learning_rate, learning_rate);
    }

    /// Sets independent rates; joint optimizers use the policy rate.
    pub fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64) {
        match &mut self.inner {
            OptimizerKind::Joint { lr, .. } => *lr = policy_learning_rate,
            OptimizerKind::Split {
                policy_lr,
                value_lr,
                ..
            } => {
                *policy_lr = policy_learning_rate;
                *value_lr = value_learning_rate;
            }
        }
    }

    fn update<P: BurnPolicy>(
        &mut self,
        model: &mut PolicyValueModel<P>,
        losses: PolicyValueLosses,
    ) {
        let value_loss = losses.value_loss.mul_scalar(losses.vf_coeff);
        match &mut self.inner {
            OptimizerKind::Joint { optimizer, lr } => {
                let grads = (losses.policy_loss + value_loss).backward();
                let grads = GradientsParams::from_grads(grads, model);
                *model = optimizer.step(*lr, model.clone(), grads);
            }
            OptimizerKind::Split {
                policy,
                policy_lr,
                value,
                value_lr,
            } => {
                let policy_grads =
                    GradientsParams::from_grads(losses.policy_loss.backward(), &model.policy);
                let value_grads =
                    GradientsParams::from_grads(value_loss.backward(), &model.value_net);
                model.policy = policy.step(*policy_lr, model.policy.clone(), policy_grads);
                model.value_net = value.step(*value_lr, model.value_net.clone(), value_grads);
            }
        }
    }

    fn to_snapshot(&self) -> OptimizerKind<OptimizerRecord> {
        match &self.inner {
            OptimizerKind::Joint { optimizer, lr } => OptimizerKind::Joint {
                optimizer: optimizer.to_record(),
                lr: *lr,
            },
            OptimizerKind::Split {
                policy,
                policy_lr,
                value,
                value_lr,
            } => OptimizerKind::Split {
                policy: policy.to_record(),
                policy_lr: *policy_lr,
                value: value.to_record(),
                value_lr: *value_lr,
            },
        }
    }

    fn load_snapshot(self, snapshot: OptimizerKind<OptimizerRecord>) -> Self {
        let inner = match (self.inner, snapshot) {
            (
                OptimizerKind::Joint { optimizer, .. },
                OptimizerKind::Joint {
                    optimizer: record,
                    lr,
                },
            ) => OptimizerKind::Joint {
                optimizer: optimizer.load_record(record),
                lr,
            },
            (
                OptimizerKind::Split { policy, value, .. },
                OptimizerKind::Split {
                    policy: policy_record,
                    policy_lr,
                    value: value_record,
                    value_lr,
                },
            ) => OptimizerKind::Split {
                policy: policy.load_record(policy_record),
                policy_lr,
                value: value.load_record(value_record),
                value_lr,
            },
            _ => panic!("snapshot optimizer layout must match the learner"),
        };
        Self { inner }
    }
}

/// Model parameters and optimizer state, independent of the policy's Rust type.
pub struct PolicyValueLearnerSnapshot {
    model: ModuleRecord,
    optimizer: OptimizerKind<OptimizerRecord>,
}

impl PolicyValueLearnerSnapshot {
    /// Saves model and optimizer records in a Burnpack snapshot file.
    ///
    /// # Panics
    /// Panics if a record cannot be serialized or the file cannot be written.
    pub fn to_file(self, file: PathBuf) {
        let model = self.model.into_bytes().unwrap();
        match self.optimizer {
            OptimizerKind::Joint { optimizer, lr } => Self::save_records(
                file,
                [
                    ("model", model),
                    ("optimizer", optimizer.into_bytes().unwrap()),
                ],
                [("lr", lr)],
            ),
            OptimizerKind::Split {
                policy,
                policy_lr,
                value,
                value_lr,
            } => Self::save_records(
                file,
                [
                    ("model", model),
                    ("policy_optimizer", policy.into_bytes().unwrap()),
                    ("value_optimizer", value.into_bytes().unwrap()),
                ],
                [("policy_lr", policy_lr), ("value_lr", value_lr)],
            ),
        }
    }

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

/// Burn policy/value learner with a joint or split optimizer.
pub struct PolicyValueLearner<P: BurnPolicy = BurnDistributionKind> {
    model: PolicyValueModel<P>,
    optimizer: PolicyValueOptimizer,
}

impl<P: BurnPolicy> PolicyValueLearner<P> {
    /// Combines a policy, value network, and optimizer, enabling training on both networks.
    ///
    /// # Panics
    /// Panics unless the value network outputs one value per observation.
    #[must_use]
    pub fn new(policy: P, value_net: NetworkKind, optimizer: PolicyValueOptimizer) -> Self {
        assert_eq!(
            value_net.output_shape().dims(),
            [1],
            "value network must output one value"
        );
        Self {
            model: PolicyValueModel { policy, value_net }.train(),
            optimizer,
        }
    }

    /// Builds a policy/value learner with a shared optimizer configuration.
    ///
    /// # Panics
    /// Panics for invalid layer widths or a value output width other than one.
    #[must_use]
    pub fn joint(
        policy: P,
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
    #[must_use]
    pub fn joint_with_network(
        policy: P,
        value_net: NetworkKind,
        optimizer_config: &AdamWConfig,
        lr: f64,
    ) -> Self {
        Self::new(
            policy,
            value_net,
            PolicyValueOptimizer::joint(optimizer_config, lr),
        )
    }

    /// Builds a policy/value learner with separate policy and value optimizers.
    ///
    /// # Panics
    /// Panics for invalid layer widths or a value output width other than one.
    #[must_use]
    pub fn split(
        policy: P,
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
    #[must_use]
    pub fn split_with_network(
        policy: P,
        value_net: NetworkKind,
        policy_optimizer_config: &AdamWConfig,
        policy_lr: f64,
        value_optimizer_config: &AdamWConfig,
        value_lr: f64,
    ) -> Self {
        Self::new(
            policy,
            value_net,
            PolicyValueOptimizer::split(
                policy_optimizer_config,
                policy_lr,
                value_optimizer_config,
                value_lr,
            ),
        )
    }

    /// Returns the current policy optimizer learning rate.
    #[must_use]
    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer.policy_learning_rate()
    }

    /// Sets the same learning rate for both policy and value updates.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.optimizer.set_learning_rate(learning_rate);
    }

    /// Captures model parameters, optimizer state, and learning rates.
    #[must_use]
    pub fn to_snapshot(&self) -> PolicyValueLearnerSnapshot {
        PolicyValueLearnerSnapshot {
            model: self.model.clone().into_record(),
            optimizer: self.optimizer.to_snapshot(),
        }
    }

    /// Restores a snapshot into matching model and optimizer configurations.
    ///
    /// # Panics
    /// Panics if the model parameters or optimizer layout do not match.
    #[must_use]
    pub fn load_snapshot(self, snapshot: PolicyValueLearnerSnapshot) -> Self {
        Self {
            model: self.model.load_record(snapshot.model),
            optimizer: self.optimizer.load_snapshot(snapshot.optimizer),
        }
    }
}

impl<P: BurnPolicy> Learner for PolicyValueLearner<P> {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        self.optimizer.update(&mut self.model, losses);
        Ok(())
    }
}

impl<P: BurnPolicy> ValueFunction for PolicyValueLearner<P> {
    type Tensor = Tensor<2>;

    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        self.model.value_net.forward(observations)
    }
}

impl<P: BurnPolicy> OnPolicyLearner for PolicyValueLearner<P> {
    type Policy = P;
    type InferencePolicy = P;

    fn inference_policy(&self) -> Self::InferencePolicy {
        self.model.policy.valid()
    }

    fn policy(&self) -> &Self::Policy {
        &self.model.policy
    }

    fn policy_learning_rate(&self) -> f64 {
        self.optimizer.policy_learning_rate()
    }

    fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64) {
        self.optimizer
            .set_learning_rates(policy_learning_rate, value_learning_rate);
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Self::Tensor> {
        Ok(Tensor::from_data(
            burn::tensor::TensorData::new(slice.to_vec(), [slice.len(), 1]),
            &self.model.value_net.devices()[0],
        ))
    }

    fn prepare_learning_tensor(t: &Self::Tensor) -> Self::Tensor {
        t.clone().detach().autodiff()
    }
}
