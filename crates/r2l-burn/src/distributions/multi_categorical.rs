use burn::{
    Tensor,
    module::Module,
    prelude::Backend,
    tensor::{
        TensorData,
        activation::{log_softmax, softmax},
    },
};
use burn_store::{ModuleStore, SafetensorsStore};
use r2l_core::{
    env::action_ranges,
    error::{Error, Result, TensorError},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    rng::with_rng,
};
use rand::distr::Distribution as RandDistribution;
use rand::distr::weighted::WeightedIndex;

use crate::sequential::Sequential;

/// Multi-categorical Burn policy for Gymnasium `MultiDiscrete` action spaces.
#[derive(Debug, Module)]
pub struct MultiCategoricalDistribution<B: Backend> {
    logits: Sequential<B>,
    nvec: Vec<usize>,
}

impl<B: Backend> MultiCategoricalDistribution<B> {
    /// Builds a multi-categorical policy network.
    #[must_use]
    pub fn build(
        observation_size: usize,
        hidden_layers: &[usize],
        nvec: Vec<usize>,
        activation: ActivationFunction,
    ) -> Self {
        let logits_size = nvec.iter().sum();
        let layers = &[&[observation_size], hidden_layers, &[logits_size]].concat();
        let logits = Sequential::build(layers, activation);
        Self { logits, nvec }
    }
}

impl<B: Backend> Actor for MultiCategoricalDistribution<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation).squeeze::<1>();
        let mut actions = Vec::new();
        for (offset, choices) in action_ranges(&self.nvec) {
            let probs: Vec<f32> = softmax(logits.clone().narrow(0, offset, choices), 0)
                .to_data()
                .to_vec()
                .map_err(|error| {
                    TensorError::operation("read multi-categorical probabilities", error)
                })?;
            let distribution = WeightedIndex::new(&probs).map_err(Error::wrap)?;
            let action = with_rng(|rng| distribution.sample(rng));
            actions.push(action as f32);
        }
        Ok(Tensor::from_data(
            TensorData::new(actions, vec![self.nvec.len()]),
            &device,
        ))
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation).squeeze::<1>();
        let actions = action_ranges(&self.nvec)
            .map(|(offset, choices)| {
                let logits: Vec<f32> = logits
                    .clone()
                    .narrow(0, offset, choices)
                    .to_data()
                    .to_vec()
                    .map_err(|error| {
                        TensorError::operation("read multi-categorical logits", error)
                    })?;
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index)
                    .ok_or_else(|| TensorError::EmptyInput {
                        operation: "select multi-categorical modal action".into(),
                    })
                    .map(|index| index as f32)
            })
            .collect::<std::result::Result<Vec<_>, TensorError>>()?;
        Ok(Tensor::from_data(
            TensorData::new(actions, vec![self.nvec.len()]),
            &device,
        ))
    }
}

impl<B: Backend> ToSafetensors for MultiCategoricalDistribution<B> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).map_err(Error::wrap)?;
        store.get_bytes().map_err(Error::wrap)
    }
}

impl<B: Backend> Policy for MultiCategoricalDistribution<B> {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        debug_assert_eq!(states.len(), actions.len());
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let logits = self.logits.forward(states);
        let mut selected_log_probs = Vec::new();
        for (action_idx, (offset, choices)) in action_ranges(&self.nvec).enumerate() {
            let logits = logits.clone().narrow(1, offset, choices);
            let log_probs = log_softmax(logits, 1);
            let action = actions.clone().narrow(1, action_idx, 1).int();
            selected_log_probs.push(log_probs.gather(1, action).squeeze_dim::<1>(1));
        }
        Ok(Tensor::stack::<2>(selected_log_probs, 0)
            .sum_dim(0)
            .squeeze())
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let logits = self.logits.forward(states);
        let mut entropies = Vec::new();
        for (offset, choices) in action_ranges(&self.nvec) {
            let logits = logits.clone().narrow(1, offset, choices);
            let probs = softmax(logits.clone(), 1);
            let log_probs = log_softmax(logits, 1);
            entropies.push((probs * log_probs).neg().sum_dim(1).squeeze_dim::<1>(1));
        }
        Ok(Tensor::stack::<2>(entropies, 0).sum_dim(0).mean())
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
