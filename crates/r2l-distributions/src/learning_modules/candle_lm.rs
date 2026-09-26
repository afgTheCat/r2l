//! Candle learners for batched policies and value networks.
//!
//! Observations and actions use `[batch, features]`; values, returns and log
//! probabilities use `[batch, 1]`. Losses must contain one element.
//! Build all parameters, including Gaussian `log_std`, through `VarMap`-backed
//! builders before constructing the optimizer. Keep the maps to save/load weights
//! with Candle's native APIs.
//!
//! Inference policies share live parameter storage with the learner and detach
//! their outputs. Collect rollouts and run optimizer updates in separate phases;
//! an inference clone is not a frozen snapshot of an earlier policy.
//!
//! ```
//! use candle_core::{DType, Device};
//! use candle_nn::{ParamsAdamW, VarBuilder, VarMap};
//! use r2l_core::models::ActivationFunction;
//! use r2l_distributions::{Categorical, networks::candle_mlp::Mlp};
//! use r2l_distributions::learning_modules::candle_lm::{
//!     PolicyValueLearner, PolicyValueOptimizer,
//! };
//!
//! let device = Device::Cpu;
//! let variables = VarMap::new();
//! let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
//! let policy = Categorical::new(Mlp::build(
//!     &[4, 8, 2], ActivationFunction::Tanh, &vb.pp("policy"),
//! )?)?;
//! // For a Gaussian, build the mean network the same way and obtain log_std
//! // from vb.pp("policy").get_with_hints((1, actions), "log_std", Init::Const(0.0)).
//! let value = Mlp::build(&[4, 8, 1], ActivationFunction::Tanh, &vb.pp("value"))?;
//! let optimizer = PolicyValueOptimizer::joint(&variables, ParamsAdamW::default(), Some(0.5))?;
//! let learner = PolicyValueLearner::new(policy, value, optimizer, device)?;
//! # Ok::<(), r2l_core::error::Error>(())
//! ```

use std::collections::HashSet;

use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use r2l_core::{
    Shape,
    error::{Error, Result},
    models::{Actor, Learner},
    on_policy::losses::FromPolicyValueLosses,
};

use crate::{
    DistributionKind, Network, OnPolicyLearner, Policy, ValueFunction, networks::candle_mlp::Mlp,
};

/// Candle distributions backed by a dense network and native variable tensors.
pub type CandleDistributionKind = DistributionKind<Mlp>;

/// Reduced policy/value losses and a value-loss multiplier.
pub struct PolicyValueLosses {
    /// Policy loss, including any entropy term.
    pub policy_loss: Tensor,
    /// Value-function loss.
    pub value_loss: Tensor,
    /// Multiplier applied to the value loss, defaulting to `1.0`.
    pub vf_coeff: f32,
}

impl PolicyValueLosses {
    /// Creates a bundle of single-element policy and value losses.
    #[must_use]
    pub fn new(policy_loss: Tensor, value_loss: Tensor) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: 1.0,
        }
    }

    /// Adds an already signed and weighted entropy term to the policy loss.
    ///
    /// # Errors
    /// Returns an error unless both terms contain one element on the same device.
    pub fn add_entropy_loss(&mut self, entropy_loss: &Tensor) -> Result<()> {
        self.policy_loss = (self.policy_loss.reshape(())? + entropy_loss.reshape(())?)?;
        Ok(())
    }

    /// Sets the value-loss multiplier applied when this bundle is optimized.
    pub fn set_vf_coeff(&mut self, vf_coeff: f32) {
        self.vf_coeff = vf_coeff;
    }
}

impl FromPolicyValueLosses<Tensor> for PolicyValueLosses {
    fn from_policy_value_losses(policy_loss: Tensor, value_loss: Tensor) -> Self {
        Self::new(policy_loss, value_loss)
    }
}

fn validate_max_norm(max_norm: Option<f32>) -> Result<()> {
    if let Some(norm) = max_norm
        && (!norm.is_finite() || norm < 0.0)
    {
        return Err(Error::invalid_parameter(
            "max_grad_norm",
            "a finite nonnegative norm",
            norm.to_string(),
        ));
    }
    Ok(())
}

fn variables(vm: &VarMap) -> Result<Vec<Var>> {
    let vars = vm.data().lock().map_err(|error| Error::InvalidState {
        operation: "read optimizer variables".into(),
        details: error.to_string(),
    })?;
    let mut named = vars.iter().collect::<Vec<_>>();
    named.sort_unstable_by_key(|(name, _)| *name);
    let mut seen = HashSet::new();
    Ok(named
        .into_iter()
        .filter(|(_, var)| seen.insert(var.id()))
        .map(|(_, var)| var.clone())
        .collect())
}

struct ClippedAdamW {
    optimizer: AdamW,
    variables: Vec<Var>,
    max_grad_norm: Option<f32>,
}

impl ClippedAdamW {
    fn new(variables: Vec<Var>, params: ParamsAdamW, max_grad_norm: Option<f32>) -> Result<Self> {
        validate_max_norm(max_grad_norm)?;
        Ok(Self {
            optimizer: AdamW::new(variables.clone(), params)?,
            variables,
            max_grad_norm,
        })
    }

    fn gradients(&self, loss: &Tensor) -> Result<GradStore> {
        let mut grads = loss.backward()?;
        if let Some(max_norm) = self.max_grad_norm {
            let mut norm_squared = 0.0f64;
            for var in &self.variables {
                if let Some(grad) = grads.get(var) {
                    norm_squared += f64::from(
                        grad.sqr()?
                            .sum_all()?
                            .to_dtype(DType::F32)?
                            .to_scalar::<f32>()?,
                    );
                }
            }
            let norm = norm_squared.sqrt();
            if norm > f64::from(max_norm) {
                let scale = f64::from(max_norm) / (norm + 1e-6);
                for var in &self.variables {
                    if let Some(grad) = grads.get(var) {
                        let clipped = (grad * scale)?;
                        grads.insert(var.as_tensor(), clipped);
                    }
                }
            }
        }
        Ok(grads)
    }
}

enum OptimizerKind {
    Joint(ClippedAdamW),
    Split {
        policy: ClippedAdamW,
        value: ClippedAdamW,
    },
}

/// Joint or split `AdamW` optimizers, optionally clipping the global gradient norm.
///
/// Each optimizer captures its variables at construction. Build networks and
/// Gaussian log standard deviations first; adding variables later does not
/// register them with an existing optimizer.
pub struct PolicyValueOptimizer {
    inner: OptimizerKind,
}

impl PolicyValueOptimizer {
    /// Builds one optimizer over all policy and value variables in `vm`.
    ///
    /// Shared parameters are updated once using the sum of both losses.
    ///
    /// # Errors
    /// Returns an error for an invalid clipping norm or failed initialization.
    pub fn joint(vm: &VarMap, params: ParamsAdamW, max_grad_norm: Option<f32>) -> Result<Self> {
        Ok(Self {
            inner: OptimizerKind::Joint(ClippedAdamW::new(variables(vm)?, params, max_grad_norm)?),
        })
    }

    /// Builds separate optimizers over disjoint policy and value variable maps.
    ///
    /// Each loss updates only its optimizer's variables, with independent `AdamW`
    /// settings and clipping norms. Use `joint` when networks share parameters.
    ///
    /// # Errors
    /// Returns an error for overlapping variables, invalid clipping norms or
    /// failed optimizer initialization.
    pub fn split(
        policy_vm: &VarMap,
        value_vm: &VarMap,
        policy_params: ParamsAdamW,
        value_params: ParamsAdamW,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    ) -> Result<Self> {
        let policy_vars = variables(policy_vm)?;
        let value_vars = variables(value_vm)?;
        let policy_ids: HashSet<_> = policy_vars.iter().map(|var| var.id()).collect();
        if value_vars.iter().any(|var| policy_ids.contains(&var.id())) {
            return Err(Error::invalid_parameter(
                "split optimizer variables",
                "disjoint policy and value variables",
                "shared variables",
            ));
        }
        Ok(Self {
            inner: OptimizerKind::Split {
                policy: ClippedAdamW::new(policy_vars, policy_params, policy_max_grad_norm)?,
                value: ClippedAdamW::new(value_vars, value_params, value_max_grad_norm)?,
            },
        })
    }

    /// Returns the current policy optimizer learning rate.
    #[must_use]
    pub fn policy_learning_rate(&self) -> f64 {
        match &self.inner {
            OptimizerKind::Joint(optimizer)
            | OptimizerKind::Split {
                policy: optimizer, ..
            } => optimizer.optimizer.learning_rate(),
        }
    }

    /// Sets the learning rate for both policy and value updates.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.set_learning_rates(learning_rate, learning_rate);
    }

    /// Sets policy and value rates independently; joint optimizers use the policy rate.
    pub fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64) {
        match &mut self.inner {
            OptimizerKind::Joint(optimizer) => {
                optimizer.optimizer.set_learning_rate(policy_learning_rate);
            }
            OptimizerKind::Split { policy, value } => {
                policy.optimizer.set_learning_rate(policy_learning_rate);
                value.optimizer.set_learning_rate(value_learning_rate);
            }
        }
    }
}

impl Learner for PolicyValueOptimizer {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        let policy_loss = losses.policy_loss.reshape(())?;
        let value_loss = (losses.value_loss.reshape(())? * f64::from(losses.vf_coeff))?;
        match &mut self.inner {
            OptimizerKind::Joint(optimizer) => {
                let grads = optimizer.gradients(&(policy_loss + value_loss)?)?;
                optimizer.optimizer.step(&grads)?;
            }
            OptimizerKind::Split { policy, value } => {
                // Backpropagate both losses before mutating any shared tensor storage.
                let policy_grads = policy.gradients(&policy_loss)?;
                let value_grads = value.gradients(&value_loss)?;
                policy.optimizer.step(&policy_grads)?;
                value.optimizer.step(&value_grads)?;
            }
        }
        Ok(())
    }
}

/// A live view of a Candle policy that detaches inputs and outputs for rollouts.
///
/// Parameter storage is shared: subsequent learner updates change this policy.
#[derive(Clone)]
pub struct InferencePolicy<P>(P);

impl<P: r2l_core::models::ToSafetensors> r2l_core::models::ToSafetensors for InferencePolicy<P> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        self.0.to_safetensors()
    }
}

impl<P: Policy<Tensor = Tensor>> Actor for InferencePolicy<P> {
    type Tensor = Tensor;

    fn action(&self, observation: Tensor) -> Result<Tensor> {
        Ok(self.0.action(observation.detach())?.detach())
    }

    fn mode_action(&self, observation: Tensor) -> Result<Tensor> {
        Ok(self.0.mode_action(observation.detach())?.detach())
    }
}

impl<P: Policy<Tensor = Tensor>> Policy for InferencePolicy<P> {
    fn action_shape(&self) -> Shape {
        self.0.action_shape()
    }

    fn log_probs(&self, observations: Tensor, actions: Tensor) -> Result<Tensor> {
        Ok(self
            .0
            .log_probs(observations.detach(), actions.detach())?
            .detach())
    }

    fn entropy(&self, observations: Tensor) -> Result<Tensor> {
        Ok(self.0.entropy(observations.detach())?.detach())
    }

    fn std(&self) -> Result<Option<f32>> {
        self.0.std()
    }
}

/// Candle on-policy learner for any batched policy and value network.
pub struct PolicyValueLearner<P = DistributionKind<Mlp>, N = Mlp> {
    policy: P,
    value_net: N,
    optimizer: PolicyValueOptimizer,
    device: Device,
}

impl<P: Policy<Tensor = Tensor> + Clone, N: Network<Tensor = Tensor>> PolicyValueLearner<P, N> {
    /// Combines a policy, value network and optimizer built from their variables.
    ///
    /// Both networks must accept the same observation width and use F32 tensors
    /// on `device`, which is also used for returns created by `tensor_from_slice`.
    /// The optimizer must contain all trainable parameters, including `log_std`.
    ///
    /// # Errors
    /// Returns an error unless the value network outputs one value per observation.
    pub fn new(
        policy: P,
        value_net: N,
        optimizer: PolicyValueOptimizer,
        device: Device,
    ) -> Result<Self> {
        if value_net.output_shape().dims() != [1] {
            return Err(Error::invalid_parameter(
                "value network output shape",
                "[1]",
                format!("{:?}", value_net.output_shape()),
            ));
        }
        Ok(Self {
            policy,
            value_net,
            optimizer,
            device,
        })
    }

    /// Returns the current policy optimizer learning rate.
    #[must_use]
    pub fn policy_learning_rate(&self) -> f64 {
        self.optimizer.policy_learning_rate()
    }
}

impl<P, N: Network<Tensor = Tensor>> ValueFunction for PolicyValueLearner<P, N> {
    type Tensor = Tensor;

    fn values(&self, observations: Tensor) -> Result<Tensor> {
        self.value_net.forward(observations)
    }
}

impl<P, N> Learner for PolicyValueLearner<P, N> {
    type Losses = PolicyValueLosses;

    fn update(&mut self, losses: Self::Losses) -> Result<()> {
        self.optimizer.update(losses)
    }
}

impl<P: Policy<Tensor = Tensor> + Clone, N: Network<Tensor = Tensor>> OnPolicyLearner
    for PolicyValueLearner<P, N>
{
    type LearningTensor = Tensor;
    type InferenceTensor = Tensor;
    type Policy = P;
    type InferencePolicy = InferencePolicy<P>;

    fn lifter(t: &Tensor) -> Tensor {
        t.detach()
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(slice, (slice.len(), 1), &self.device)?)
    }

    fn inference_policy(&self) -> Self::InferencePolicy {
        InferencePolicy(self.policy.clone())
    }

    fn policy(&self) -> &P {
        &self.policy
    }

    fn set_learning_rates(&mut self, policy_learning_rate: f64, value_learning_rate: f64) {
        self.optimizer
            .set_learning_rates(policy_learning_rate, value_learning_rate);
    }
}
