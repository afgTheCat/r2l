//! Burn policy distributions used by the on-policy stack.
//!
//! This module exposes concrete policy implementations for discrete and
//! continuous action spaces together with [`crate::distributions::PolicyKind`],
//! an enum that erases the concrete policy type behind one Burn-facing policy
//! interface.

use burn::{Tensor, module::Module, prelude::Backend};
use r2l_core::{
    env::ActionSpaceType,
    models::{ActivationFunction, Actor, Policy},
};

use crate::distributions::{
    categorical_distribution::CategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution,
};
/// Categorical policy distribution for discrete action spaces.
pub mod categorical_distribution;
/// Diagonal-Gaussian policy distribution for continuous action spaces.
pub mod diagonal_distribution;
/// Recurrent categorical policy distribution for discrete action spaces.
pub mod recurrent_categorical_distribution;

/// Erased Burn policy type covering the supported action-space variants.
///
/// This enum is the main policy type used by the Burn on-policy learning
/// modules. It dispatches to a categorical policy for discrete action spaces
/// and to a diagonal-Gaussian policy for continuous action spaces.
#[derive(Debug, Module)]
pub enum PolicyKind<B: Backend> {
    /// Policy for discrete action spaces.
    Categorical(CategoricalDistribution<B>),
    /// Policy for continuous action spaces.
    Diag(DiagGaussianDistribution<B>),
}

impl<B: Backend> PolicyKind<B> {
    fn categorical(policy_layers: &[usize], activation: ActivationFunction) -> Self {
        PolicyKind::Categorical(CategoricalDistribution::<B>::build(
            policy_layers,
            activation,
        ))
    }

    fn continuous(policy_layers: &[usize], activation: ActivationFunction) -> Self {
        PolicyKind::Diag(DiagGaussianDistribution::build(policy_layers, activation))
    }

    /// Builds the appropriate Burn policy for the given action-space type.
    pub fn build(
        action_space_type: ActionSpaceType,
        policy_layers: &[usize],
        activation: ActivationFunction,
    ) -> Self {
        match action_space_type {
            ActionSpaceType::Discrete => Self::categorical(policy_layers, activation),
            ActionSpaceType::Continuous => Self::continuous(policy_layers, activation),
        }
    }
}

impl<B: Backend> Actor for PolicyKind<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.action(observation),
            Self::Diag(diag) => diag.action(observation),
        }
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        match self {
            Self::Categorical(cat) => cat.try_serialize(),
            Self::Diag(diag) => diag.try_serialize(),
        }
    }
}

impl<B: Backend> Policy for PolicyKind<B> {
    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(observations, actions),
            Self::Diag(diag) => diag.log_probs(observations, actions),
        }
    }

    fn std(&self) -> anyhow::Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::Diag(diag) => diag.std(),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::Diag(diag) => diag.entropy(states),
        }
    }

    fn resample_noise(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::Diag(diag) => diag.resample_noise(),
        }
    }
}
