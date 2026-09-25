//! Burn policy distributions used by the on-policy stack.
//!
//! This module exposes concrete policy implementations for discrete and
//! Box action spaces together with [`crate::distributions::BurnPolicyKind`],
//! an enum that erases the concrete policy type behind one Burn-facing policy
//! interface.

use burn::{Tensor, module::Module, prelude::Backend};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use r2l_core::Shape;
use r2l_core::{
    env::Space,
    error::{Error, Result},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    networks::{MlpConfig, NetworkConfig},
    tensor::R2lTensor,
};

use crate::{
    distributions::{
        bernoulli::MultiBernoulliDistribution, categorical::CategoricalDistribution,
        composite::CompositeDistribution, diagonal::DiagGaussianDistribution,
        multi_categorical::MultiCategoricalDistribution,
    },
    networks::NetworkKind,
};

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

/// Erased Burn policy type covering the supported action-space variants.
///
/// This enum is the main policy type used by the Burn on-policy learning
/// modules. It dispatches to a categorical policy for discrete action spaces
/// and to a diagonal-Gaussian policy for Box action spaces.
#[derive(Debug, Module)]
pub enum BurnDistributionKind<B: Backend> {
    /// Policy for discrete action spaces.
    Categorical(CategoricalDistribution<B, NetworkKind<B>>),
    /// Policy for Box action spaces.
    Diag(DiagGaussianDistribution<B, NetworkKind<B>>),
    /// Policy for multi-discrete action spaces.
    MultiCategorical(MultiCategoricalDistribution<B, NetworkKind<B>>),
    /// Policy for multi-binary action spaces.
    MultiBernoulli(MultiBernoulliDistribution<B, NetworkKind<B>>),
    /// Policy for tuple and dict action spaces.
    Composite(CompositeDistribution<B>),
}

impl<B: Backend> BurnDistributionKind<B> {
    /// Loads policy parameters from safetensors bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored policy parameters cannot be loaded.
    pub fn load_from_bytes(mut self, bytes: Vec<u8>) -> Result<Self> {
        let mut store = SafetensorsStore::from_bytes(Some(bytes));
        match &mut self {
            Self::Categorical(policy) => policy.load_from(&mut store).map_err(Error::wrap)?,
            Self::Diag(policy) => policy.load_from(&mut store).map_err(Error::wrap)?,
            Self::MultiCategorical(policy) => policy.load_from(&mut store).map_err(Error::wrap)?,
            Self::MultiBernoulli(policy) => policy.load_from(&mut store).map_err(Error::wrap)?,
            Self::Composite(policy) => policy.load_from(&mut store).map_err(Error::wrap)?,
        };
        Ok(self)
    }

    /// Builds the appropriate Burn policy for the given action space.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer configuration or action space is unsupported.
    pub fn build<T: R2lTensor>(
        action_space: Space<T>,
        policy_layers: &[usize],
        activation: ActivationFunction,
        log_std_init: f32,
    ) -> Result<Self> {
        if policy_layers.len() < 2 {
            return Err(Error::invalid_parameter(
                "policy_layers",
                "at least an input and output layer",
                format!("{policy_layers:?}"),
            ));
        }
        let config = NetworkConfig::Mlp(MlpConfig {
            hidden_layers: policy_layers[1..policy_layers.len() - 1].to_vec(),
            activation,
        });
        Self::build_with_network(
            action_space,
            &Shape::from([policy_layers[0]]),
            &config,
            log_std_init,
        )
    }

    /// Builds an action distribution using the selected network architecture.
    ///
    /// # Errors
    ///
    /// Returns an error if the action space or network dimensions are invalid.
    pub fn build_with_network<T: R2lTensor>(
        action_space: Space<T>,
        observation_shape: &Shape,
        config: &NetworkConfig,
        log_std_init: f32,
    ) -> Result<Self> {
        let network = |output_size| {
            NetworkKind::build(config, observation_shape, output_size, &Default::default())
        };
        Ok(match action_space {
            Space::Discrete(choices) => {
                Self::Categorical(CategoricalDistribution::from_network(network(choices)?))
            }
            Space::Box { shape, .. } => {
                let size = shape.iter().product();
                Self::Diag(DiagGaussianDistribution::from_network(
                    network(size)?,
                    size,
                    log_std_init,
                ))
            }
            Space::MultiDiscrete { nvec, .. } => {
                let nvec: Vec<usize> = nvec.to_vec()?.into_iter().map(|n| n as usize).collect();
                if nvec.contains(&0) {
                    return Err(NetworkKind::<B>::invalid_config(
                        "category counts must be positive",
                    ));
                }
                Self::MultiCategorical(MultiCategoricalDistribution::from_network(
                    network(nvec.iter().sum())?,
                    nvec,
                ))
            }
            Space::MultiBinary { shape } => {
                let size = shape.iter().product();
                Self::MultiBernoulli(MultiBernoulliDistribution::from_network(
                    network(size)?,
                    size,
                ))
            }
            Space::Tuple(spaces) => Self::Composite(CompositeDistribution::build_with_network(
                spaces,
                observation_shape,
                config,
                log_std_init,
            )?),
            Space::Dict(spaces) => Self::Composite(CompositeDistribution::build_with_network(
                spaces.into_values().collect(),
                observation_shape,
                config,
                log_std_init,
            )?),
        })
    }
}

impl<B: Backend> Actor for BurnDistributionKind<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.action(observation),
            Self::Diag(diag) => diag.action(observation),
            Self::MultiCategorical(multi) => multi.action(observation),
            Self::MultiBernoulli(bernoulli) => bernoulli.action(observation),
            Self::Composite(composite) => composite.action(observation),
        }
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.mode_action(observation),
            Self::Diag(diag) => diag.mode_action(observation),
            Self::MultiCategorical(multi) => multi.mode_action(observation),
            Self::MultiBernoulli(bernoulli) => bernoulli.mode_action(observation),
            Self::Composite(composite) => composite.mode_action(observation),
        }
    }
}

impl<B: Backend> ToSafetensors for BurnDistributionKind<B> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        match self {
            Self::Categorical(cat) => cat.to_safetensors(),
            Self::Diag(diag) => diag.to_safetensors(),
            Self::MultiCategorical(multi) => multi.to_safetensors(),
            Self::MultiBernoulli(bernoulli) => bernoulli.to_safetensors(),
            Self::Composite(composite) => composite.to_safetensors(),
        }
    }
}

impl<B: Backend> Policy for BurnDistributionKind<B> {
    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(observations, actions),
            Self::Diag(diag) => diag.log_probs(observations, actions),
            Self::MultiCategorical(multi) => multi.log_probs(observations, actions),
            Self::MultiBernoulli(bernoulli) => bernoulli.log_probs(observations, actions),
            Self::Composite(composite) => composite.log_probs(observations, actions),
        }
    }

    fn std(&self) -> Result<Option<f32>> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::Diag(diag) => diag.std(),
            Self::MultiCategorical(multi) => multi.std(),
            Self::MultiBernoulli(bernoulli) => bernoulli.std(),
            Self::Composite(composite) => composite.std(),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::Diag(diag) => diag.entropy(states),
            Self::MultiCategorical(multi) => multi.entropy(states),
            Self::MultiBernoulli(bernoulli) => bernoulli.entropy(states),
            Self::Composite(composite) => composite.entropy(states),
        }
    }
}
