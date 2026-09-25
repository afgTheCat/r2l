use burn::{Tensor, module::Module, prelude::Backend};
use burn_store::{ModuleStore, SafetensorsStore};
use r2l_core::Shape;
use r2l_core::{
    env::Space,
    error::{Error, Result, TensorError},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    networks::{MlpConfig, NetworkConfig},
    tensor::R2lTensor,
};

use crate::{
    distributions::{
        BurnDistributionKind, bernoulli::MultiBernoulliDistribution,
        categorical::CategoricalDistribution, diagonal::DiagGaussianDistribution,
        multi_categorical::MultiCategoricalDistribution,
    },
    networks::NetworkKind,
};

#[derive(Debug, Module)]
enum CompositePolicyChildren<B: Backend> {
    Categorical(CategoricalDistribution<B, NetworkKind<B>>),
    Diag(DiagGaussianDistribution<B, NetworkKind<B>>),
    MultiCategorical(MultiCategoricalDistribution<B, NetworkKind<B>>),
    MultiBernoulli(MultiBernoulliDistribution<B, NetworkKind<B>>),
}

impl<B: Backend> CompositePolicyChildren<B> {
    fn action(&self, observation: Tensor<B, 1>) -> Result<Tensor<B, 1>> {
        match self {
            Self::Categorical(policy) => policy.action(observation),
            Self::Diag(policy) => policy.action(observation),
            Self::MultiCategorical(policy) => policy.action(observation),
            Self::MultiBernoulli(policy) => policy.action(observation),
        }
    }

    fn mode_action(&self, observation: Tensor<B, 1>) -> Result<Tensor<B, 1>> {
        match self {
            Self::Categorical(policy) => policy.mode_action(observation),
            Self::Diag(policy) => policy.mode_action(observation),
            Self::MultiCategorical(policy) => policy.mode_action(observation),
            Self::MultiBernoulli(policy) => policy.mode_action(observation),
        }
    }

    fn log_probs(&self, states: &[Tensor<B, 1>], actions: &[Tensor<B, 1>]) -> Result<Tensor<B, 1>> {
        match self {
            Self::Categorical(policy) => policy.log_probs(states, actions),
            Self::Diag(policy) => policy.log_probs(states, actions),
            Self::MultiCategorical(policy) => policy.log_probs(states, actions),
            Self::MultiBernoulli(policy) => policy.log_probs(states, actions),
        }
    }

    fn entropy(&self, states: &[Tensor<B, 1>]) -> Result<Tensor<B, 1>> {
        match self {
            Self::Categorical(policy) => policy.entropy(states),
            Self::Diag(policy) => policy.entropy(states),
            Self::MultiCategorical(policy) => policy.entropy(states),
            Self::MultiBernoulli(policy) => policy.entropy(states),
        }
    }
}

/// Composite Burn policy for tuple and dict action spaces.
#[derive(Debug, Module)]
pub struct CompositeDistribution<B: Backend> {
    policies: Vec<CompositePolicyChildren<B>>,
    action_sizes: Vec<usize>,
}

impl<B: Backend> CompositeDistribution<B> {
    /// Builds one child policy per nested action space.
    ///
    /// # Errors
    ///
    /// Returns an error if an action space is unsupported or a child policy cannot be built.
    pub fn build<T: R2lTensor>(
        action_spaces: Vec<Space<T>>,
        policy_layers: &[usize],
        activation: ActivationFunction,
        log_std_init: f32,
    ) -> Result<Self> {
        if policy_layers.len() < 2 {
            return Err(NetworkKind::<B>::invalid_config(
                "expected input and output layer widths",
            ));
        }
        let config = NetworkConfig::Mlp(MlpConfig {
            hidden_layers: policy_layers[1..policy_layers.len() - 1].to_vec(),
            activation,
        });
        Self::build_with_network(
            action_spaces,
            &Shape::from([policy_layers[0]]),
            &config,
            log_std_init,
        )
    }

    pub(crate) fn build_with_network<T: R2lTensor>(
        action_spaces: Vec<Space<T>>,
        observation_shape: &Shape,
        config: &NetworkConfig,
        log_std_init: f32,
    ) -> Result<Self> {
        let mut policies = Vec::new();
        let mut action_sizes = Vec::new();
        for action_space in action_spaces {
            Self::push_child(
                action_space,
                observation_shape,
                config,
                log_std_init,
                &mut policies,
                &mut action_sizes,
            )?;
        }
        if policies.is_empty() {
            return Err(TensorError::EmptyInput {
                operation: "build composite policy".into(),
            }
            .into());
        }
        Ok(Self {
            policies,
            action_sizes,
        })
    }

    fn push_child<T: R2lTensor>(
        action_space: Space<T>,
        observation_shape: &Shape,
        config: &NetworkConfig,
        log_std_init: f32,
        policies: &mut Vec<CompositePolicyChildren<B>>,
        action_sizes: &mut Vec<usize>,
    ) -> Result<()> {
        let action_size = action_space.action_size();
        let children = match action_space {
            Space::Tuple(spaces) => spaces,
            Space::Dict(spaces) => spaces.into_values().collect(),
            space => {
                let policy = BurnDistributionKind::build_with_network(
                    space,
                    observation_shape,
                    config,
                    log_std_init,
                )?;
                policies.push(match policy {
                    BurnDistributionKind::Categorical(policy) => {
                        CompositePolicyChildren::Categorical(policy)
                    }
                    BurnDistributionKind::Diag(policy) => CompositePolicyChildren::Diag(policy),
                    BurnDistributionKind::MultiCategorical(policy) => {
                        CompositePolicyChildren::MultiCategorical(policy)
                    }
                    BurnDistributionKind::MultiBernoulli(policy) => {
                        CompositePolicyChildren::MultiBernoulli(policy)
                    }
                    BurnDistributionKind::Composite(_) => unreachable!(),
                });
                action_sizes.push(action_size);
                return Ok(());
            }
        };
        for child in children {
            Self::push_child(
                child,
                observation_shape,
                config,
                log_std_init,
                policies,
                action_sizes,
            )?;
        }
        Ok(())
    }
}

impl<B: Backend> Actor for CompositeDistribution<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let mut actions = Vec::new();
        for policy in &self.policies {
            actions.push(policy.action(observation.clone())?);
        }
        Ok(Tensor::cat(actions, 0))
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let mut actions = Vec::new();
        for policy in &self.policies {
            actions.push(policy.mode_action(observation.clone())?);
        }
        Ok(Tensor::cat(actions, 0))
    }
}

impl<B: Backend> ToSafetensors for CompositeDistribution<B> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).map_err(Error::wrap)?;
        store.get_bytes().map_err(Error::wrap)
    }
}

impl<B: Backend> Policy for CompositeDistribution<B> {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        debug_assert_eq!(states.len(), actions.len());
        let mut offset = 0;
        let mut log_probs = Vec::new();
        for (policy, action_size) in self.policies.iter().zip(&self.action_sizes) {
            let child_actions: Vec<_> = actions
                .iter()
                .map(|action| action.clone().narrow(0, offset, *action_size))
                .collect();
            log_probs.push(policy.log_probs(states, &child_actions)?);
            offset += action_size;
        }
        Ok(Tensor::stack::<2>(log_probs, 0).sum_dim(0).squeeze_dim(0))
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        let mut entropies = Vec::new();
        for policy in &self.policies {
            entropies.push(policy.entropy(states)?);
        }
        Ok(Tensor::stack::<2>(entropies, 0).sum_dim(0).squeeze_dim(0))
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
