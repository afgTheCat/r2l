//! Candle policy distributions used by the on-policy stack.
//!
//! This module exposes concrete policy implementations for discrete and
//! Box action spaces together with [`crate::distributions::CandlePolicyKind`],
//! an enum that erases the concrete policy type behind one Candle-facing policy
//! interface.

/// Bernoulli policy distribution for multi-binary action spaces.
pub mod bernoulli;
/// Categorical policy distribution for discrete action spaces.
pub mod categorical;
/// Composite policy distribution for tuple and dict action spaces.
pub mod composite;
/// Diagonal-Gaussian policy distribution for Box action spaces.
pub mod diagonal;
/// Multi-categorical policy distribution for multi-discrete action spaces.
pub mod multi_categorical;

use std::{f32, fmt::Debug};

use anyhow::Result;
use bernoulli::BernoulliDistribution;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use categorical::CategoricalDistribution;
use composite::CompositeDistribution;
use diagonal::DiagGaussianDistribution;
use multi_categorical::MultiCategoricalDistribution;
use r2l_core::{
    env::Space,
    models::{ActivationFunction, Actor, Policy, PolicyMetadata},
    tensor::R2lTensor,
};
use safetensors::SafeTensors;

/// Erased Candle policy type covering the supported action-space variants.
///
/// This enum is the main policy type used by the Candle on-policy learning
/// modules. It dispatches to a categorical policy for discrete action spaces
/// and to a diagonal-Gaussian policy for Box action spaces.
#[derive(Debug, Clone)]
pub enum CandlePolicyKind {
    /// Policy for discrete action spaces.
    Categorical(CategoricalDistribution),
    /// Policy for Box action spaces.
    DiagGaussian(DiagGaussianDistribution),
    /// Policy for multi-discrete action spaces.
    MultiCategorical(MultiCategoricalDistribution),
    /// Policy for multi-binary action spaces.
    Bernoulli(BernoulliDistribution),
    /// Policy for tuple and dict action spaces.
    Composite(CompositeDistribution),
}

impl CandlePolicyKind {
    /// Returns the Candle device used by the underlying policy.
    pub fn device(&self) -> Device {
        match self {
            Self::Categorical(c) => c.device(),
            Self::DiagGaussian(d) => d.device(),
            Self::MultiCategorical(m) => m.device(),
            Self::Bernoulli(b) => b.device(),
            Self::Composite(c) => c.device(),
        }
    }

    /// Returns the flattened observation size expected by the policy.
    pub fn observation_size(&self) -> usize {
        match self {
            Self::Categorical(c) => c.observation_size(),
            Self::DiagGaussian(d) => d.observation_size(),
            Self::MultiCategorical(m) => m.observation_size(),
            Self::Bernoulli(b) => b.observation_size(),
            Self::Composite(c) => c.observation_size(),
        }
    }

    /// Builds a Candle policy from serialized safetensors bytes.
    pub fn from_bytes(bytes: &[u8], device: Device) -> Self {
        let (_, safe_tensors_metadata) = SafeTensors::read_metadata(bytes).unwrap();
        let metadata = PolicyMetadata::from_safetensors_metadata(
            safe_tensors_metadata.metadata().as_ref().unwrap(),
        );
        let tensors = candle_core::safetensors::load_buffer(bytes, &device).unwrap();

        if tensors.contains_key("policy.log_std") {
            Self::DiagGaussian(DiagGaussianDistribution::from_parts(
                tensors, device, metadata,
            ))
        } else {
            Self::Categorical(CategoricalDistribution::from_parts(
                tensors, device, metadata,
            ))
        }
    }

    /// Builds the appropriate Candle policy for the given action space.
    pub fn build<T: R2lTensor>(
        action_space: Space<T>,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        observation_size: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        Self::build_with_prefix(
            action_space,
            policy_varbuilder,
            hidden_layers,
            observation_size,
            activation,
            "policy",
        )
    }

    pub(crate) fn build_with_prefix<T: R2lTensor>(
        action_space: Space<T>,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        observation_size: usize,
        activation: ActivationFunction,
        prefix: &str,
    ) -> Result<Self> {
        match action_space {
            Space::Discrete(size) => {
                let layers = &[hidden_layers, &[size]].concat();
                Ok(Self::Categorical(CategoricalDistribution::build(
                    observation_size,
                    size,
                    layers,
                    policy_varbuilder,
                    policy_varbuilder.device().clone(),
                    prefix,
                    activation,
                )?))
            }
            Space::Box { shape, .. } => {
                let size = shape.iter().product();
                let layers = &[hidden_layers, &[size]].concat();
                let log_std = policy_varbuilder.get(size, &format!("{prefix}.log_std"))?;
                Ok(Self::DiagGaussian(DiagGaussianDistribution::build(
                    observation_size,
                    layers,
                    policy_varbuilder,
                    log_std,
                    prefix,
                    activation,
                )?))
            }
            Space::MultiDiscrete { nvec, .. } => {
                Ok(Self::MultiCategorical(MultiCategoricalDistribution::build(
                    observation_size,
                    nvec.to_vec().into_iter().map(|n| n as usize).collect(),
                    hidden_layers,
                    policy_varbuilder,
                    policy_varbuilder.device().clone(),
                    prefix,
                    activation,
                )?))
            }
            Space::MultiBinary { shape } => {
                let size = shape.iter().product();
                Ok(Self::Bernoulli(BernoulliDistribution::build(
                    observation_size,
                    size,
                    hidden_layers,
                    policy_varbuilder,
                    policy_varbuilder.device().clone(),
                    prefix,
                    activation,
                )?))
            }
            Space::Tuple(spaces) => Ok(Self::Composite(CompositeDistribution::build(
                spaces,
                policy_varbuilder,
                hidden_layers,
                observation_size,
                activation,
                prefix,
            )?)),
            Space::Dict(spaces) => Ok(Self::Composite(CompositeDistribution::build(
                spaces.into_values().collect(),
                policy_varbuilder,
                hidden_layers,
                observation_size,
                activation,
                prefix,
            )?)),
        }
    }
}

impl Actor for CandlePolicyKind {
    type Tensor = Tensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.action(observation),
            Self::DiagGaussian(diag) => diag.action(observation),
            Self::MultiCategorical(multi) => multi.action(observation),
            Self::Bernoulli(bernoulli) => bernoulli.action(observation),
            Self::Composite(composite) => composite.action(observation),
        }
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        match self {
            Self::Categorical(cat) => cat.try_serialize(),
            Self::DiagGaussian(diag) => diag.try_serialize(),
            Self::MultiCategorical(multi) => multi.try_serialize(),
            Self::Bernoulli(bernoulli) => bernoulli.try_serialize(),
            Self::Composite(composite) => composite.try_serialize(),
        }
    }
}

impl Policy for CandlePolicyKind {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(states, actions),
            Self::DiagGaussian(diag) => diag.log_probs(states, actions),
            Self::MultiCategorical(multi) => multi.log_probs(states, actions),
            Self::Bernoulli(bernoulli) => bernoulli.log_probs(states, actions),
            Self::Composite(composite) => composite.log_probs(states, actions),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::DiagGaussian(diag) => diag.entropy(states),
            Self::MultiCategorical(multi) => multi.entropy(states),
            Self::Bernoulli(bernoulli) => bernoulli.entropy(states),
            Self::Composite(composite) => composite.entropy(states),
        }
    }

    fn std(&self) -> Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::DiagGaussian(diag) => diag.std(),
            Self::MultiCategorical(multi) => multi.std(),
            Self::Bernoulli(bernoulli) => bernoulli.std(),
            Self::Composite(composite) => composite.std(),
        }
    }

    fn resample_noise(&mut self) -> Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::DiagGaussian(diag) => diag.resample_noise(),
            Self::MultiCategorical(multi) => multi.resample_noise(),
            Self::Bernoulli(bernoulli) => bernoulli.resample_noise(),
            Self::Composite(composite) => composite.resample_noise(),
        }
    }
}
