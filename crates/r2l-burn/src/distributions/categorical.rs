use burn::{
    module::Module,
    prelude::Backend,
    tensor::{
        Tensor, TensorData,
        activation::{log_softmax, softmax},
    },
};
use burn_store::{ModuleSnapshot, ModuleStore, SafetensorsStore};
use itertools::Itertools;
use r2l_core::{
    error::{Error, InvalidParameterError, Result, TensorError},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    rng::with_rng,
};
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

use crate::sequential::Sequential;

/// Categorical Burn policy for discrete action spaces.
///
/// This policy produces category indices sampled from logits predicted by a
/// feed-forward network and implements the `r2l-core` [`Actor`] and [`Policy`]
/// traits.
#[derive(Debug, Module)]
pub struct CategoricalDistribution<B: Backend> {
    logits: Sequential<B>,
}

impl<B: Backend> CategoricalDistribution<B> {
    /// Builds a categorical policy network.
    ///
    /// # Errors
    ///
    /// Returns an error if `logits_layers` is empty.
    pub fn build(logits_layers: &[usize], activation: ActivationFunction) -> Result<Self> {
        if logits_layers.is_empty() {
            return Err(Error::InvalidParameter(Box::new(
                InvalidParameterError::InvalidValue {
                    name: "logits_layers".into(),
                    expected: "at least one layer".into(),
                    value: "[]".into(),
                },
            )));
        }
        let logits: Sequential<B> = Sequential::build(logits_layers, activation);
        Ok(Self { logits })
    }

    /// Builds a categoriacal policy using a safetensor store
    ///
    /// # Errors
    ///
    /// Returns an error if the stored network dimensions or parameters are invalid.
    pub fn from_store(store: &mut SafetensorsStore) -> Result<Self> {
        let logits_layers = Sequential::<B>::dims_from_store("logits", store);
        let mut distribution = Self::build(&logits_layers, ActivationFunction::default())?;
        distribution.load_from(store).map_err(Error::wrap)?;
        Ok(distribution)
    }
}

impl<B: Backend> Actor for CategoricalDistribution<B> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation);
        let action_probs: Vec<f32> = softmax(logits, 1)
            .to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("read categorical probabilities", error))?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = with_rng(|rng| distribution.sample(rng));
        let action = Tensor::from_data(TensorData::new(vec![action as f32], vec![1]), &device);
        Ok(action)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits: Vec<f32> = self
            .logits
            .forward(observation)
            .to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("read categorical logits", error))?;
        let action = logits
            .iter()
            .position_max_by(|a, b| a.total_cmp(b))
            .ok_or_else(|| TensorError::EmptyInput {
                operation: "select categorical modal action".into(),
            })?;
        let action = Tensor::from_data(TensorData::new(vec![action as f32], vec![1]), &device);
        Ok(action)
    }
}

impl<B: Backend> ToSafetensors for CategoricalDistribution<B> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store
            .collect_from(self)
            .map_err(r2l_core::error::Error::wrap)?;
        store.get_bytes().map_err(r2l_core::error::Error::wrap)
    }
}

impl<B: Backend> Policy for CategoricalDistribution<B> {
    // FIXME: check the other fixme comment for DiagGaussian
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        debug_assert_eq!(states.len(), actions.len());
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let logits = self.logits.forward(states);
        let log_probs = log_softmax(logits, 1);
        Ok(log_probs.gather(1, actions.int()).squeeze_dim::<1>(1))
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let logits = self.logits.forward(states);
        let probs = softmax(logits.clone(), 1);
        let log_probs = log_softmax(logits, 1);
        let entropy_per_state = (probs * log_probs).neg().sum_dim(1);
        let entropy = entropy_per_state.mean();
        Ok(entropy)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
