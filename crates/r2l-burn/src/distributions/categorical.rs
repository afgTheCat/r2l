use anyhow::bail;
use burn::{
    module::Module,
    prelude::Backend,
    tensor::{
        Tensor, TensorData,
        activation::{log_softmax, softmax},
    },
};
use burn_store::{ModuleSnapshot, ModuleStore, SafetensorsStore};
use r2l_core::{
    models::{ActivationFunction, Actor, Policy},
    rng::with_rng,
};
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

use crate::sequential::Sequential;

/// Categorical Burn policy for discrete action spaces.
///
/// This policy produces one-hot actions sampled from logits predicted by a
/// feed-forward network and implements the `r2l-core` [`Actor`] and [`Policy`]
/// traits.
#[derive(Debug, Module)]
pub struct CategoricalDistribution<B: Backend> {
    logits: Sequential<B>,
    action_size: usize,
}

impl<B: Backend> CategoricalDistribution<B> {
    /// Builds a categorical policy network.
    pub fn build(logits_layers: &[usize], activation: ActivationFunction) -> Self {
        let action_size = *logits_layers.last().unwrap();
        let logits: Sequential<B> = Sequential::build(logits_layers, activation);
        Self {
            logits,
            action_size,
        }
    }

    // TODO: this is quite brittle like this
    /// Builds a categoriacal policy using a safetensor store
    pub fn from_store(store: &mut SafetensorsStore) -> Self {
        let logits_layers = Sequential::<B>::dims_from_store("logits", store);
        let mut distribution = Self::build(&logits_layers, ActivationFunction::default());
        distribution
            .load_from(store)
            .expect("failed to load CategoricalDistribution from store");
        distribution
    }
}

impl<B: Backend> Actor for CategoricalDistribution<B> {
    type Tensor = Tensor<B, 1>;
    type State = ();

    fn action(
        &self,
        observation: Self::Tensor,
        _state: Option<Self::State>,
    ) -> anyhow::Result<(Self::Tensor, Self::State)> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation);
        let action_probs: Vec<f32> = softmax(logits, 1).to_data().to_vec().unwrap();
        let distribution = WeightedIndex::new(&action_probs).unwrap();
        let action = with_rng(|rng| distribution.sample(rng));
        let mut action_mask: Vec<f32> = vec![0.0; self.action_size];
        action_mask[action] = 1.;
        Ok((
            Tensor::from_data(
                TensorData::new(action_mask, vec![self.action_size]),
                &device,
            ),
            (),
        ))
    }

    // This will serialize the model to safetesnors
    fn try_serialize(&self) -> Option<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).unwrap();
        store.get_bytes().ok()
    }
}

impl<B: Backend> Policy for CategoricalDistribution<B> {
    // FIXME: check the other fixme comment for DiagGaussian
    fn log_probs(
        &self,
        states: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> anyhow::Result<Self::Tensor> {
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let logits = self.logits.forward(states);
        let log_probs = log_softmax(logits, 1);
        let log_probs = (actions * log_probs).sum_dim(1);
        Ok(log_probs.squeeze())
    }

    fn entropy(&self, states: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let states: Tensor<B, 2> = Tensor::stack(states.to_vec(), 0);
        let logits = self.logits.forward(states);
        let probs = softmax(logits.clone(), 1);
        let log_probs = log_softmax(logits, 1);
        let entropy_per_state = (probs * log_probs).neg().sum_dim(1);
        let entropy = entropy_per_state.mean();
        Ok(entropy)
    }

    fn std(&self) -> anyhow::Result<f32> {
        bail!("standard deviation is not defined for categorical distributions")
    }
}
