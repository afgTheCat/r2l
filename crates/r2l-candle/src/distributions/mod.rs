//! Candle policy distributions used by the on-policy stack.
//!
//! This module exposes concrete policy implementations for discrete and
//! continuous action spaces together with [`crate::distributions::CandlePolicyKind`],
//! an enum that erases the concrete policy type behind one Candle-facing policy
//! interface.

/// Bernoulli policy distribution for multi-binary action spaces.
pub mod bernoulli_distribution;
/// Categorical policy distribution for discrete action spaces.
pub mod categorical_distribution;
/// Diagonal-Gaussian policy distribution for continuous action spaces.
pub mod diagonal_distribution;
/// Multi-categorical policy distribution for multi-discrete action spaces.
pub mod multi_categorical_distribution;

use std::{f32, fmt::Debug};

use anyhow::Result;
use bernoulli_distribution::BernoulliDistribution;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use categorical_distribution::CategoricalDistribution;
use diagonal_distribution::DiagGaussianDistribution;
use multi_categorical_distribution::MultiCategoricalDistribution;
use r2l_core::{
    env::ActionSpaceType,
    models::{ActivationFunction, Actor, Policy, PolicyMetadata},
};
use safetensors::SafeTensors;

/// Erased Candle policy type covering the supported action-space variants.
///
/// This enum is the main policy type used by the Candle on-policy learning
/// modules. It dispatches to a categorical policy for discrete action spaces
/// and to a diagonal-Gaussian policy for continuous action spaces.
#[derive(Debug, Clone)]
pub enum CandlePolicyKind {
    /// Policy for discrete action spaces.
    Categorical(CategoricalDistribution),
    /// Policy for continuous action spaces.
    DiagGaussian(DiagGaussianDistribution),
    /// Policy for multi-discrete action spaces.
    MultiCategorical(MultiCategoricalDistribution),
    /// Policy for multi-binary action spaces.
    Bernoulli(BernoulliDistribution),
}

impl CandlePolicyKind {
    /// Returns the Candle device used by the underlying policy.
    pub fn device(&self) -> Device {
        match self {
            Self::Categorical(c) => c.device(),
            Self::DiagGaussian(d) => d.device(),
            Self::MultiCategorical(m) => m.device(),
            Self::Bernoulli(b) => b.device(),
        }
    }

    /// Returns the flattened observation size expected by the policy.
    pub fn observation_size(&self) -> usize {
        match self {
            Self::Categorical(c) => c.observation_size(),
            Self::DiagGaussian(d) => d.observation_size(),
            Self::MultiCategorical(m) => m.observation_size(),
            Self::Bernoulli(b) => b.observation_size(),
        }
    }

    /// Builds a Candle policy from serialized safetensors bytes.
    pub fn from_bytes(bytes: &[u8], device: Device) -> Self {
        let (_, safe_tensors_metadata) = SafeTensors::read_metadata(bytes).unwrap();
        let metadata = PolicyMetadata::from_safetensors_metadata(
            safe_tensors_metadata.metadata().as_ref().unwrap(),
        );
        let tensors = candle_core::safetensors::load_buffer(bytes, &device).unwrap();

        if tensors.contains_key("log_std") {
            Self::DiagGaussian(DiagGaussianDistribution::from_parts(
                tensors, device, metadata,
            ))
        } else {
            Self::Categorical(CategoricalDistribution::from_parts(
                tensors, device, metadata,
            ))
        }
    }

    /// Builds a categorical Candle policy.
    pub fn categorical(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let layers = &[hidden_layers, &[action_size]].concat();
        let distr = CategoricalDistribution::build(
            observation_size,
            action_size,
            layers,
            policy_varbuilder,
            policy_varbuilder.device().clone(),
            "policy",
            activation,
        )?;
        Ok(Self::Categorical(distr))
    }

    /// Builds a diagonal-Gaussian Candle policy.
    pub fn diag_gaussian(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let layers = &[hidden_layers, &[action_size]].concat();
        let log_std = policy_varbuilder.get(action_size, "log_std")?;
        let distr = DiagGaussianDistribution::build(
            observation_size,
            layers,
            policy_varbuilder,
            log_std,
            "policy",
            activation,
        )?;
        Ok(Self::DiagGaussian(distr))
    }

    /// Builds a multi-categorical Candle policy.
    pub fn multi_categorical(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        nvec: Vec<usize>,
        observation_size: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let distr = MultiCategoricalDistribution::build(
            observation_size,
            nvec,
            hidden_layers,
            policy_varbuilder,
            policy_varbuilder.device().clone(),
            "policy",
            activation,
        )?;
        Ok(Self::MultiCategorical(distr))
    }

    /// Builds a Bernoulli Candle policy.
    pub fn bernoulli(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let distr = BernoulliDistribution::build(
            observation_size,
            action_size,
            hidden_layers,
            policy_varbuilder,
            policy_varbuilder.device().clone(),
            "policy",
            activation,
        )?;
        Ok(Self::Bernoulli(distr))
    }

    /// Builds the appropriate Candle policy for the given action-space type.
    pub fn build(
        action_space: ActionSpaceType,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        match action_space {
            ActionSpaceType::Discrete => Self::categorical(
                policy_varbuilder,
                hidden_layers,
                action_size,
                observation_size,
                activation,
            ),
            ActionSpaceType::Continuous => Self::diag_gaussian(
                policy_varbuilder,
                hidden_layers,
                action_size,
                observation_size,
                activation,
            ),
            ActionSpaceType::MultiDiscrete { nvec } => Self::multi_categorical(
                policy_varbuilder,
                hidden_layers,
                nvec,
                observation_size,
                activation,
            ),
            ActionSpaceType::MultiBinary { size } => Self::bernoulli(
                policy_varbuilder,
                hidden_layers,
                size,
                observation_size,
                activation,
            ),
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
        }
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        match self {
            Self::Categorical(cat) => cat.try_serialize(),
            Self::DiagGaussian(diag) => diag.try_serialize(),
            Self::MultiCategorical(multi) => multi.try_serialize(),
            Self::Bernoulli(bernoulli) => bernoulli.try_serialize(),
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
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::DiagGaussian(diag) => diag.entropy(states),
            Self::MultiCategorical(multi) => multi.entropy(states),
            Self::Bernoulli(bernoulli) => bernoulli.entropy(states),
        }
    }

    fn std(&self) -> Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::DiagGaussian(diag) => diag.std(),
            Self::MultiCategorical(multi) => multi.std(),
            Self::Bernoulli(bernoulli) => bernoulli.std(),
        }
    }

    fn resample_noise(&mut self) -> Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::DiagGaussian(diag) => diag.resample_noise(),
            Self::MultiCategorical(multi) => multi.resample_noise(),
            Self::Bernoulli(bernoulli) => bernoulli.resample_noise(),
        }
    }
}
