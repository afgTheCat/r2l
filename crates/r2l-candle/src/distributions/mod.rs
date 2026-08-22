//! Candle policy distributions used by the on-policy stack.
//!
//! This module exposes concrete policy implementations for discrete and
//! Box action spaces together with [`crate::distributions::CandlePolicyKind`],
//! an enum that erases the concrete policy type behind one Candle-facing policy
//! interface.

/// Multi-Bernoulli policy distribution for multi-binary action spaces.
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

use bernoulli::MultiBernoulliDistribution;
use candle_core::{Device, Tensor};
use candle_nn::{Init, VarBuilder};
use categorical::CategoricalDistribution;
use composite::CompositeDistribution;
use diagonal::DiagGaussianDistribution;
use multi_categorical::MultiCategoricalDistribution;
use r2l_core::{
    env::Space,
    error::Result,
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    tensor::R2lTensor,
};

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
    MultiBernoulli(MultiBernoulliDistribution),
    /// Policy for tuple and dict action spaces.
    Composite(CompositeDistribution),
}

impl CandlePolicyKind {
    /// Returns the Candle device used by the underlying policy.
    #[must_use]
    pub fn device(&self) -> Device {
        match self {
            Self::Categorical(c) => c.device(),
            Self::DiagGaussian(d) => d.device(),
            Self::MultiCategorical(m) => m.device(),
            Self::MultiBernoulli(b) => b.device(),
            Self::Composite(c) => c.device(),
        }
    }

    /// Returns the flattened observation size expected by the policy.
    #[must_use]
    pub fn observation_size(&self) -> usize {
        match self {
            Self::Categorical(c) => c.observation_size(),
            Self::DiagGaussian(d) => d.observation_size(),
            Self::MultiCategorical(m) => m.observation_size(),
            Self::MultiBernoulli(b) => b.observation_size(),
            Self::Composite(c) => c.observation_size(),
        }
    }

    /// Builds the appropriate Candle policy for the given action space.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected policy cannot be built.
    pub fn build<T: R2lTensor>(
        action_space: Space<T>,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        observation_size: usize,
        activation: ActivationFunction,
        log_std_init: f32,
    ) -> Result<Self> {
        Self::build_with_prefix(
            action_space,
            policy_varbuilder,
            hidden_layers,
            observation_size,
            activation,
            log_std_init,
            "policy",
        )
    }

    pub(crate) fn build_with_prefix<T: R2lTensor>(
        action_space: Space<T>,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        observation_size: usize,
        activation: ActivationFunction,
        log_std_init: f32,
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
                let log_std = policy_varbuilder.get_with_hints(
                    size,
                    &format!("{prefix}.log_std"),
                    Init::Const(f64::from(log_std_init)),
                )?;
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
                let nvec = nvec
                    .to_vec()
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                Ok(Self::MultiCategorical(MultiCategoricalDistribution::build(
                    observation_size,
                    nvec.into_iter().map(|n| n as usize).collect(),
                    hidden_layers,
                    policy_varbuilder,
                    policy_varbuilder.device().clone(),
                    prefix,
                    activation,
                )?))
            }
            Space::MultiBinary { shape } => {
                let size = shape.iter().product();
                Ok(Self::MultiBernoulli(MultiBernoulliDistribution::build(
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
                log_std_init,
                prefix,
            )?)),
            Space::Dict(spaces) => Ok(Self::Composite(CompositeDistribution::build(
                spaces.into_values().collect(),
                policy_varbuilder,
                hidden_layers,
                observation_size,
                activation,
                log_std_init,
                prefix,
            )?)),
        }
    }

    pub(crate) fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            Self::Categorical(policy) => policy.named_tensors(prefix),
            Self::DiagGaussian(policy) => policy.named_tensors(prefix),
            Self::MultiCategorical(policy) => policy.named_tensors(prefix),
            Self::MultiBernoulli(policy) => policy.named_tensors(prefix),
            Self::Composite(policy) => policy.named_tensors(prefix),
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
            Self::MultiBernoulli(bernoulli) => bernoulli.action(observation),
            Self::Composite(composite) => composite.action(observation),
        }
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.mode_action(observation),
            Self::DiagGaussian(diag) => diag.mode_action(observation),
            Self::MultiCategorical(multi) => multi.mode_action(observation),
            Self::MultiBernoulli(bernoulli) => bernoulli.mode_action(observation),
            Self::Composite(composite) => composite.mode_action(observation),
        }
    }
}

impl ToSafetensors for CandlePolicyKind {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        match self {
            Self::Categorical(cat) => cat.to_safetensors(),
            Self::DiagGaussian(diag) => diag.to_safetensors(),
            Self::MultiCategorical(multi) => multi.to_safetensors(),
            Self::MultiBernoulli(bernoulli) => bernoulli.to_safetensors(),
            Self::Composite(composite) => composite.to_safetensors(),
        }
    }
}

impl Policy for CandlePolicyKind {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(states, actions),
            Self::DiagGaussian(diag) => diag.log_probs(states, actions),
            Self::MultiCategorical(multi) => multi.log_probs(states, actions),
            Self::MultiBernoulli(bernoulli) => bernoulli.log_probs(states, actions),
            Self::Composite(composite) => composite.log_probs(states, actions),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::DiagGaussian(diag) => diag.entropy(states),
            Self::MultiCategorical(multi) => multi.entropy(states),
            Self::MultiBernoulli(bernoulli) => bernoulli.entropy(states),
            Self::Composite(composite) => composite.entropy(states),
        }
    }

    fn std(&self) -> Result<Option<f32>> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::DiagGaussian(diag) => diag.std(),
            Self::MultiCategorical(multi) => multi.std(),
            Self::MultiBernoulli(bernoulli) => bernoulli.std(),
            Self::Composite(composite) => composite.std(),
        }
    }
}
