//! Candle policy distributions used by the on-policy stack.
//!
//! This module exposes concrete policy implementations for discrete and
//! continuous action spaces together with [`crate::distributions::CandlePolicyKind`],
//! an enum that erases the concrete policy type behind one Candle-facing policy
//! interface.

/// Categorical policy distribution for discrete action spaces.
pub mod categorical_distribution;
/// Diagonal-Gaussian policy distribution for continuous action spaces.
pub mod diagonal_distribution;

use std::{f32, fmt::Debug};

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use categorical_distribution::CategoricalDistribution;
use diagonal_distribution::DiagGaussianDistribution;
use r2l_core::{
    env::ActionSpaceType,
    models::{Actor, Policy},
};

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
}

impl CandlePolicyKind {
    /// Returns the Candle device used by the underlying policy.
    pub fn device(&self) -> Device {
        match self {
            Self::Categorical(c) => c.device(),
            Self::DiagGaussian(d) => d.device(),
        }
    }

    /// Returns the flattened observation size expected by the policy.
    pub fn observation_size(&self) -> usize {
        match self {
            Self::Categorical(c) => c.observation_size(),
            Self::DiagGaussian(d) => d.observation_size(),
        }
    }

    /// Builds a categorical Candle policy.
    pub fn categorical(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
    ) -> Result<Self> {
        let layers = &[hidden_layers, &[action_size]].concat();
        let distr = CategoricalDistribution::build(
            observation_size,
            action_size,
            layers,
            policy_varbuilder,
            policy_varbuilder.device().clone(),
            "policy",
        )?;
        Ok(Self::Categorical(distr))
    }

    /// Builds a diagonal-Gaussian Candle policy.
    pub fn diag_gaussian(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
    ) -> Result<Self> {
        let layers = &[hidden_layers, &[action_size]].concat();
        let log_std = policy_varbuilder.get(action_size, "log_std")?;
        let distr = DiagGaussianDistribution::build(
            observation_size,
            layers,
            policy_varbuilder,
            log_std,
            "policy",
        )?;
        Ok(Self::DiagGaussian(distr))
    }

    /// Builds the appropriate Candle policy for the given action-space type.
    pub fn build(
        action_space: ActionSpaceType,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
    ) -> Result<Self> {
        match action_space {
            ActionSpaceType::Discrete => Self::categorical(
                policy_varbuilder,
                hidden_layers,
                action_size,
                observation_size,
            ),
            ActionSpaceType::Continuous => Self::diag_gaussian(
                policy_varbuilder,
                hidden_layers,
                action_size,
                observation_size,
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
        }
    }
}

impl Policy for CandlePolicyKind {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(states, actions),
            Self::DiagGaussian(diag) => diag.log_probs(states, actions),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::DiagGaussian(diag) => diag.entropy(states),
        }
    }

    fn std(&self) -> Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::DiagGaussian(diag) => diag.std(),
        }
    }

    fn resample_noise(&mut self) -> Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::DiagGaussian(diag) => diag.resample_noise(),
        }
    }
}
