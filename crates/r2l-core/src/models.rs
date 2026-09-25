use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{error::Result, tensor::R2lTensor};

/// Activation function used between hidden layers in feed-forward networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// Exponential linear unit activation with the backend default alpha.
    Elu,
    /// Gaussian error linear unit activation.
    Gelu,
    /// Gaussian error linear unit activation using the backend tanh approximation.
    GeluApproximate,
    /// Hard sigmoid activation with backend default parameters.
    HardSigmoid,
    /// Hard swish activation.
    HardSwish,
    /// Leaky rectified linear unit activation with the backend default slope.
    LeakyRelu,
    /// Rectified linear unit activation.
    Relu,
    /// Sigmoid activation.
    Sigmoid,
    /// Hyperbolic tangent activation.
    #[default]
    Tanh,
}

impl fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Elu => "elu",
            Self::Gelu => "gelu",
            Self::GeluApproximate => "gelu_approximate",
            Self::HardSigmoid => "hard_sigmoid",
            Self::HardSwish => "hard_swish",
            Self::LeakyRelu => "leaky_relu",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
        };
        f.write_str(name)
    }
}

impl FromStr for ActivationFunction {
    type Err = String;

    fn from_str(name: &str) -> std::result::Result<Self, Self::Err> {
        match name {
            "elu" => Ok(Self::Elu),
            "gelu" => Ok(Self::Gelu),
            "gelu_approximate" => Ok(Self::GeluApproximate),
            "hard_sigmoid" => Ok(Self::HardSigmoid),
            "hard_swish" => Ok(Self::HardSwish),
            "leaky_relu" => Ok(Self::LeakyRelu),
            "relu" => Ok(Self::Relu),
            "sigmoid" => Ok(Self::Sigmoid),
            "tanh" => Ok(Self::Tanh),
            _ => Err(format!("unknown activation function: {name}")),
        }
    }
}

/// A policy-like object that can choose an action for one observation.
///
/// Actors are the inference-time surface used by samplers. They must be
/// sendable so rollout collection can move them into worker threads.
/// Built-in policies accept one flattened observation row `[1, features]` and
/// return `[1, actions]`; `ActorWrapper` adapts these rows to environment tensors.
pub trait Actor: Send + 'static {
    /// Tensor type accepted as observations and returned as actions.
    type Tensor: R2lTensor;

    /// Selects an action for a single observation.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference fails.
    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor>;

    /// Selects the modal action for a single observation without sampling.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference fails.
    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor>;
}

/// A policy that can be serialized as a safetensors artifact.
pub trait ToSafetensors {
    /// Serializes this policy as safetensors bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy parameters cannot be serialized.
    fn to_safetensors(&self) -> Result<Vec<u8>>;
}

/// Trainable action distribution interface used by on-policy algorithms.
///
/// Observations and actions are rank-two batches. Log probabilities have shape
/// `[batch, 1]`; entropy is the batch mean of the summed action-component entropies.
/// Actions supplied for training must lie in the policy's support.
///
/// A `Policy` extends [`Actor`] with the quantities needed to compute policy
/// gradient losses and entropy bonuses over a batch.
pub trait Policy: Actor {
    /// Shape of one flattened action, without the batch axis.
    fn action_shape(&self) -> crate::Shape;

    /// Computes log probabilities for batched observation/action pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy cannot evaluate the batch.
    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor>;

    /// Returns a representative action standard deviation when available.
    ///
    /// # Errors
    ///
    /// Returns an error if the standard deviation cannot be computed.
    /// Returns `Ok(None)` when the policy has no meaningful scalar standard deviation.
    fn std(&self) -> Result<Option<f32>>;

    /// Computes the policy entropy for a batch of states.
    ///
    /// # Errors
    ///
    /// Returns an error if the entropy cannot be computed.
    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor>;
}

/// Component that learns from backend-specific loss values.
pub trait Learner {
    /// Loss bundle consumed by this module.
    type Losses;

    /// Applies one optimization update from precomputed losses.
    ///
    /// # Errors
    ///
    /// Returns an error if the optimizer update fails.
    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

/// Value function over rank-two observation batches.
pub trait ValueFunction {
    /// Tensor type used for observations and values.
    type Tensor: R2lTensor;

    /// Returns `[batch, 1]` values for `[batch, features]` observations.
    ///
    /// # Errors
    /// Returns an error for incompatible input dimensions or failed evaluation.
    fn values(&self, observations: Self::Tensor) -> Result<Self::Tensor>;
}
