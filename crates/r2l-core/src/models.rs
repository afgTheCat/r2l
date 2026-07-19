use std::{collections::HashMap, fmt, str::FromStr};

use anyhow::Result;

use crate::tensor::R2lTensor;

/// Activation function used between hidden layers in feed-forward networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

/// Metadata stored next to policy tensors in a safetensors archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyMetadata {
    /// Hidden-layer activation function.
    pub activation: ActivationFunction,
}

impl PolicyMetadata {
    /// Converts the metadata into the string map accepted by safetensors.
    pub fn to_safetensors_metadata(&self) -> HashMap<String, String> {
        HashMap::from([("activation".to_string(), self.activation.to_string())])
    }

    /// Builds policy metadata from the string map stored by safetensors.
    pub fn from_safetensors_metadata(metadata: &HashMap<String, String>) -> Self {
        Self {
            activation: metadata.get("activation").unwrap().parse().unwrap(),
        }
    }
}

/// A policy-like object that can choose an action for one observation.
///
/// Actors are the inference-time surface used by samplers. They must be
/// sendable so rollout collection can move them into worker threads.
pub trait Actor: Send + 'static {
    /// Tensor type accepted as observations and returned as actions.
    type Tensor: R2lTensor;

    /// State carried between action selections.
    ///
    /// Stateless actors use `()`.
    type State: Clone + Send + Sync + 'static;

    /// Selects an action and returns the state for the next invocation.
    ///
    /// `None` indicates that the observation starts a new episode.
    fn action(
        &self,
        observation: Self::Tensor,
        state: Option<Self::State>,
    ) -> Result<(Self::Tensor, Self::State)>;

    /// Tries to serialize the Actor
    fn try_serialize(&self) -> Option<Vec<u8>> {
        None
    }
}

/// Trainable action distribution interface used by on-policy algorithms.
///
/// A `Policy` extends [`Actor`] with the quantities needed to compute policy
/// gradient losses and entropy bonuses over a batch.
pub trait Policy: Actor {
    /// Computes log probabilities for batched observation/action pairs.
    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor>;

    /// Returns a representative action standard deviation when available.
    fn std(&self) -> Result<f32>;

    /// Computes the policy entropy for a batch of states.
    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor>;

    /// Resamples exploration noise for policies that use state-independent
    /// noise. Implementations without such noise may keep the default no-op.
    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Results produced by evaluating one contiguous recurrent-policy sequence.
pub struct RecurrentPolicyOutput<T, S> {
    /// Log probability of each supplied action.
    pub log_probs: T,
    /// Entropy of the action distribution at each step.
    pub entropy: T,
    /// State produced after the final observation in the sequence.
    pub final_state: S,
}

/// Trainable policy interface for contiguous recurrent sequences.
///
/// The initial state is supplied only once. Implementations must carry the
/// resulting state through the sequence so gradients can flow across steps.
pub trait RecurrentPolicy: Actor {
    /// Evaluates observations and actions starting from `initial_state`.
    fn evaluate_sequence(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
        initial_state: Option<&Self::State>,
    ) -> Result<RecurrentPolicyOutput<Self::Tensor, Self::State>>;

    /// Returns a representative action standard deviation when available.
    fn std(&self) -> Result<f32>;

    /// Resamples exploration noise for policies that use state-independent
    /// noise. Implementations without such noise may keep the default no-op.
    fn resample_noise(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Component that applies backend-specific optimizer updates.
pub trait LearningModule {
    /// Loss bundle consumed by this module.
    type Losses;

    /// Applies one optimization update from precomputed losses.
    fn update(&mut self, losses: Self::Losses) -> Result<()>;
}

/// Batched value-function interface.
pub trait ValueFunction {
    /// Tensor type used for observations and returned values.
    type Tensor: Clone;

    /// Estimates values for a batch of observations.
    fn values(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor>;
}
