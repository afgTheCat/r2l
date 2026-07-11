use anyhow::bail;
use burn::{Tensor, module::Module, prelude::Backend};
use burn_store::{ModuleStore, SafetensorsStore};
use r2l_core::{
    env::Space,
    models::{ActivationFunction, Actor, Policy},
    tensor::R2lTensor,
};

use crate::distributions::{
    bernoulli::BernoulliDistribution, categorical::CategoricalDistribution,
    diagonal::DiagGaussianDistribution, multi_categorical::MultiCategoricalDistribution,
};

#[derive(Debug, Module)]
enum CompositePolicyChildren<B: Backend> {
    Categorical(CategoricalDistribution<B>),
    Diag(DiagGaussianDistribution<B>),
    MultiCategorical(MultiCategoricalDistribution<B>),
    Bernoulli(BernoulliDistribution<B>),
}

impl<B: Backend> CompositePolicyChildren<B> {
    fn action(&self, observation: Tensor<B, 1>) -> anyhow::Result<Tensor<B, 1>> {
        match self {
            Self::Categorical(policy) => policy.action(observation),
            Self::Diag(policy) => policy.action(observation),
            Self::MultiCategorical(policy) => policy.action(observation),
            Self::Bernoulli(policy) => policy.action(observation),
        }
    }

    fn log_probs(
        &self,
        states: &[Tensor<B, 1>],
        actions: &[Tensor<B, 1>],
    ) -> anyhow::Result<Tensor<B, 1>> {
        match self {
            Self::Categorical(policy) => policy.log_probs(states, actions),
            Self::Diag(policy) => policy.log_probs(states, actions),
            Self::MultiCategorical(policy) => policy.log_probs(states, actions),
            Self::Bernoulli(policy) => policy.log_probs(states, actions),
        }
    }

    fn entropy(&self, states: &[Tensor<B, 1>]) -> anyhow::Result<Tensor<B, 1>> {
        match self {
            Self::Categorical(policy) => policy.entropy(states),
            Self::Diag(policy) => policy.entropy(states),
            Self::MultiCategorical(policy) => policy.entropy(states),
            Self::Bernoulli(policy) => policy.entropy(states),
        }
    }

    fn resample_noise(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Categorical(policy) => policy.resample_noise(),
            Self::Diag(policy) => policy.resample_noise(),
            Self::MultiCategorical(policy) => policy.resample_noise(),
            Self::Bernoulli(policy) => policy.resample_noise(),
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
    pub fn build<T: R2lTensor>(
        action_spaces: Vec<Space<T>>,
        policy_layers: &[usize],
        activation: ActivationFunction,
    ) -> Self {
        let mut policies = Vec::new();
        let mut action_sizes = Vec::new();
        for action_space in action_spaces {
            Self::push_child(
                action_space,
                policy_layers,
                activation,
                &mut policies,
                &mut action_sizes,
            );
        }
        Self {
            policies,
            action_sizes,
        }
    }

    fn push_child<T: R2lTensor>(
        action_space: Space<T>,
        policy_layers: &[usize],
        activation: ActivationFunction,
        policies: &mut Vec<CompositePolicyChildren<B>>,
        action_sizes: &mut Vec<usize>,
    ) {
        let action_size = action_space.size();
        match action_space {
            Space::Discrete(_) => {
                let child_layers = [
                    &[policy_layers[0]],
                    &policy_layers[1..policy_layers.len() - 1],
                    &[action_size],
                ]
                .concat();
                policies.push(CompositePolicyChildren::Categorical(
                    CategoricalDistribution::build(&child_layers, activation),
                ));
                action_sizes.push(action_size);
            }
            Space::Box { .. } => {
                let child_layers = [
                    &[policy_layers[0]],
                    &policy_layers[1..policy_layers.len() - 1],
                    &[action_size],
                ]
                .concat();
                policies.push(CompositePolicyChildren::Diag(
                    DiagGaussianDistribution::build(&child_layers, activation),
                ));
                action_sizes.push(action_size);
            }
            Space::MultiDiscrete { nvec, .. } => {
                policies.push(CompositePolicyChildren::MultiCategorical(
                    MultiCategoricalDistribution::build(
                        policy_layers[0],
                        &policy_layers[1..policy_layers.len() - 1],
                        nvec.to_vec().into_iter().map(|n| n as usize).collect(),
                        activation,
                    ),
                ));
                action_sizes.push(action_size);
            }
            Space::MultiBinary { .. } => {
                policies.push(CompositePolicyChildren::Bernoulli(
                    BernoulliDistribution::build(
                        policy_layers[0],
                        &policy_layers[1..policy_layers.len() - 1],
                        action_size,
                        activation,
                    ),
                ));
                action_sizes.push(action_size);
            }
            Space::Tuple(spaces) => {
                for space in spaces {
                    Self::push_child(space, policy_layers, activation, policies, action_sizes);
                }
            }
            Space::Dict(spaces) => {
                for space in spaces.into_values() {
                    Self::push_child(space, policy_layers, activation, policies, action_sizes);
                }
            }
        }
    }
}

impl<B: Backend> Actor for CompositeDistribution<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        let mut actions = Vec::new();
        for policy in &self.policies {
            actions.push(policy.action(observation.clone())?);
        }
        Ok(Tensor::cat(actions, 0))
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).unwrap();
        store.get_bytes().ok()
    }
}

impl<B: Backend> Policy for CompositeDistribution<B> {
    fn log_probs(
        &self,
        states: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> anyhow::Result<Self::Tensor> {
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
        Ok(Tensor::stack::<2>(log_probs, 0).sum_dim(0).squeeze())
    }

    fn entropy(&self, states: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let mut entropies = Vec::new();
        for policy in &self.policies {
            entropies.push(policy.entropy(states)?);
        }
        Ok(Tensor::stack::<2>(entropies, 0).sum_dim(0).squeeze())
    }

    fn std(&self) -> anyhow::Result<f32> {
        bail!("standard deviation is not defined for composite distributions")
    }

    fn resample_noise(&mut self) -> anyhow::Result<()> {
        for policy in &mut self.policies {
            policy.resample_noise()?;
        }
        Ok(())
    }
}
