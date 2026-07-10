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
    bernoulli_distribution::BernoulliDistribution,
    categorical_distribution::CategoricalDistribution,
    composite_distribution::CompositeDistribution, diagonal_distribution::DiagGaussianDistribution,
    multi_categorical_distribution::MultiCategoricalDistribution,
};
/// Bernoulli policy distribution for multi-binary action spaces.
pub mod bernoulli_distribution;
/// Categorical policy distribution for discrete action spaces.
pub mod categorical_distribution;
/// Composite policy distribution for tuple and dict action spaces.
pub mod composite_distribution;
/// Diagonal-Gaussian policy distribution for continuous action spaces.
pub mod diagonal_distribution;
/// Multi-categorical policy distribution for multi-discrete action spaces.
pub mod multi_categorical_distribution;
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
    /// Policy for multi-discrete action spaces.
    MultiCategorical(MultiCategoricalDistribution<B>),
    /// Policy for multi-binary action spaces.
    Bernoulli(BernoulliDistribution<B>),
    /// Policy for tuple and dict action spaces.
    Composite(CompositeDistribution<B>),
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

    fn multi_categorical(
        policy_layers: &[usize],
        nvec: Vec<usize>,
        activation: ActivationFunction,
    ) -> Self {
        PolicyKind::MultiCategorical(MultiCategoricalDistribution::build(
            policy_layers[0],
            &policy_layers[1..policy_layers.len() - 1],
            nvec,
            activation,
        ))
    }

    fn bernoulli(
        policy_layers: &[usize],
        action_size: usize,
        activation: ActivationFunction,
    ) -> Self {
        PolicyKind::Bernoulli(BernoulliDistribution::build(
            policy_layers[0],
            &policy_layers[1..policy_layers.len() - 1],
            action_size,
            activation,
        ))
    }

    /// Builds the appropriate Burn policy for the given action-space type.
    pub fn build(
        action_space_type: ActionSpaceType,
        policy_layers: &[usize],
        activation: ActivationFunction,
    ) -> Self {
        match action_space_type {
            ActionSpaceType::Discrete { .. } => Self::categorical(policy_layers, activation),
            ActionSpaceType::Continuous { .. } => Self::continuous(policy_layers, activation),
            ActionSpaceType::MultiDiscrete { nvec } => {
                Self::multi_categorical(policy_layers, nvec, activation)
            }
            ActionSpaceType::MultiBinary { size } => {
                Self::bernoulli(policy_layers, size, activation)
            }
            ActionSpaceType::Tuple(spaces) => PolicyKind::Composite(CompositeDistribution::build(
                spaces,
                policy_layers,
                activation,
            )),
            ActionSpaceType::Dict(spaces) => PolicyKind::Composite(CompositeDistribution::build(
                spaces.into_values().collect(),
                policy_layers,
                activation,
            )),
        }
    }
}

impl<B: Backend> Actor for PolicyKind<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.action(observation),
            Self::Diag(diag) => diag.action(observation),
            Self::MultiCategorical(multi) => multi.action(observation),
            Self::Bernoulli(bernoulli) => bernoulli.action(observation),
            Self::Composite(composite) => composite.action(observation),
        }
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        match self {
            Self::Categorical(cat) => cat.try_serialize(),
            Self::Diag(diag) => diag.try_serialize(),
            Self::MultiCategorical(multi) => multi.try_serialize(),
            Self::Bernoulli(bernoulli) => bernoulli.try_serialize(),
            Self::Composite(composite) => composite.try_serialize(),
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
            Self::MultiCategorical(multi) => multi.log_probs(observations, actions),
            Self::Bernoulli(bernoulli) => bernoulli.log_probs(observations, actions),
            Self::Composite(composite) => composite.log_probs(observations, actions),
        }
    }

    fn std(&self) -> anyhow::Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::Diag(diag) => diag.std(),
            Self::MultiCategorical(multi) => multi.std(),
            Self::Bernoulli(bernoulli) => bernoulli.std(),
            Self::Composite(composite) => composite.std(),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::Diag(diag) => diag.entropy(states),
            Self::MultiCategorical(multi) => multi.entropy(states),
            Self::Bernoulli(bernoulli) => bernoulli.entropy(states),
            Self::Composite(composite) => composite.entropy(states),
        }
    }

    fn resample_noise(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::Diag(diag) => diag.resample_noise(),
            Self::MultiCategorical(multi) => multi.resample_noise(),
            Self::Bernoulli(bernoulli) => bernoulli.resample_noise(),
            Self::Composite(composite) => composite.resample_noise(),
        }
    }
}
