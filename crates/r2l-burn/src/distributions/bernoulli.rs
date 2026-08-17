use burn::{
    Tensor,
    module::Module,
    prelude::Backend,
    tensor::{TensorData, activation::sigmoid},
};
use burn_store::{ModuleStore, SafetensorsStore};
use r2l_core::{
    error::{Error, Result, TensorError},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    rng::with_rng,
};

use crate::sequential::Sequential;

/// Independent Bernoulli policy for Gymnasium `MultiBinary` action spaces.
///
/// This is equivalent to a multi-categorical policy with two categories per
/// action component, but uses one logit per component instead of two.
#[derive(Debug, Module)]
pub struct MultiBernoulliDistribution<B: Backend> {
    logits: Sequential<B>,
    action_size: usize,
}

impl<B: Backend> MultiBernoulliDistribution<B> {
    /// Builds a multi-Bernoulli policy network.
    #[must_use]
    pub fn build(
        observation_size: usize,
        hidden_layers: &[usize],
        action_size: usize,
        activation: ActivationFunction,
    ) -> Self {
        let layers = &[&[observation_size], hidden_layers, &[action_size]].concat();
        let logits = Sequential::build(layers, activation);
        Self {
            logits,
            action_size,
        }
    }
}

impl<B: Backend> Actor for MultiBernoulliDistribution<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation).squeeze::<1>();
        let probs: Vec<f32> = sigmoid(logits)
            .to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("read multi-Bernoulli probabilities", error))?;
        let actions = probs
            .into_iter()
            .map(|prob| {
                if with_rng(rand::RngExt::random::<f32>) < prob {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        Ok(Tensor::from_data(
            TensorData::new(actions, vec![self.action_size]),
            &device,
        ))
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation).squeeze::<1>();
        let probs: Vec<f32> = sigmoid(logits)
            .to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("read multi-Bernoulli probabilities", error))?;
        let actions = probs
            .into_iter()
            .map(|probability| if probability >= 0.5 { 1.0 } else { 0.0 })
            .collect();
        Ok(Tensor::from_data(
            TensorData::new(actions, vec![self.action_size]),
            &device,
        ))
    }
}

impl<B: Backend> ToSafetensors for MultiBernoulliDistribution<B> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).map_err(Error::wrap)?;
        store.get_bytes().map_err(Error::wrap)
    }
}

impl<B: Backend> Policy for MultiBernoulliDistribution<B> {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        debug_assert_eq!(states.len(), actions.len());
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let probs = sigmoid(self.logits.forward(states)).clamp(1e-6, 1. - 1e-6);
        let ones = probs.ones_like();
        let log_probs =
            actions.clone() * probs.clone().log() + (ones.clone() - actions) * (ones - probs).log();
        Ok(log_probs.sum_dim(1).squeeze())
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let probs = sigmoid(self.logits.forward(states)).clamp(1e-6, 1. - 1e-6);
        let ones = probs.ones_like();
        let entropy_per_bit = probs.clone() * probs.clone().log()
            + (ones.clone() - probs.clone()) * (ones - probs).log();
        Ok(entropy_per_bit.neg().sum_dim(1).mean())
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
