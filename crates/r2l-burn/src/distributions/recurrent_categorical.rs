use anyhow::{bail, ensure};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Rnn, RnnConfig, RnnState},
    tensor::{
        Tensor, TensorData,
        activation::{log_softmax, softmax},
        backend::Backend,
    },
};
use burn_store::{ModuleSnapshot, ModuleStore, SafetensorsStore};
use r2l_core::{
    models::{ActivationFunction, Actor, RecurrentPolicy, RecurrentPolicyOutput},
    rng::with_rng,
};
use rand::distr::Distribution as RandDistribution;
use rand::distr::weighted::WeightedIndex;

use crate::sequential::Sequential;

/// Recurrent categorical Burn policy for discrete action spaces.
///
/// Action selection accepts and returns hidden state. The [`RecurrentPolicy`]
/// implementation evaluates a contiguous sequence from a supplied state.
#[derive(Debug, Module)]
pub struct RecurrentCategoricalDistribution<B: Backend> {
    encoder: Sequential<B>,
    recurrent: Rnn<B>,
    logits: Linear<B>,
    action_size: usize,
}

impl<B: Backend> RecurrentCategoricalDistribution<B> {
    /// Builds a recurrent categorical policy network.
    ///
    /// `layers` follows the same convention as the feed-forward policies:
    /// first observation size, optional hidden encoder sizes, final action size.
    /// The recurrent hidden size is the encoder output size.
    pub fn build(layers: &[usize]) -> Self {
        assert!(
            layers.len() >= 2,
            "recurrent categorical policy requires input and output sizes"
        );
        let device = Default::default();
        let observation_size = layers[0];
        let action_size = *layers.last().unwrap();
        let recurrent_size = if layers.len() > 2 {
            layers[layers.len() - 2]
        } else {
            observation_size
        };
        let encoder_layers = if layers.len() > 2 {
            layers[..layers.len() - 1].to_vec()
        } else {
            vec![observation_size]
        };
        let encoder = Sequential::build(&encoder_layers, ActivationFunction::default());
        let recurrent = RnnConfig::new(recurrent_size, recurrent_size, true).init::<B>(&device);
        let logits = LinearConfig::new(recurrent_size, action_size)
            .with_bias(true)
            .init::<B>(&device);
        Self {
            encoder,
            recurrent,
            logits,
            action_size,
        }
    }

    fn sequence_logits(
        &self,
        states: Tensor<B, 2>,
        state: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let encoded = self.encoder.forward(states);
        let sequence = encoded.unsqueeze_dim(0);
        let state = state.map(RnnState::new);
        let (recurrent_output, state) = self.recurrent.forward(sequence, state);
        let recurrent_output = recurrent_output.squeeze_dim(0);
        (self.logits.forward(recurrent_output), state.hidden)
    }

    // TODO: this is brittle in the same way as the other Burn distributions.
    /// Builds a recurrent categorical policy using a safetensor store.
    pub fn from_store(store: &mut SafetensorsStore) -> Self {
        let mut encoder_layers = Sequential::<B>::dims_from_store("encoder", store);
        let action_size = store
            .get_snapshot("logits.weight")
            .expect("failed to read recurrent categorical logits weight from store")
            .expect("missing recurrent categorical logits weight in store")
            .shape
            .dims::<2>()[1];
        encoder_layers.push(action_size);
        let mut distribution = Self::build(&encoder_layers);
        distribution
            .load_from(store)
            .expect("failed to load RecurrentCategoricalDistribution from store");
        distribution
    }
}

impl<B: Backend> Actor for RecurrentCategoricalDistribution<B> {
    type Tensor = Tensor<B, 1>;
    type State = Tensor<B, 2>;

    fn action(
        &self,
        observation: Self::Tensor,
        state: Option<Self::State>,
    ) -> anyhow::Result<(Self::Tensor, Self::State)> {
        let device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let (logits, state) = self.sequence_logits(observation, state);
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
            state,
        ))
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).unwrap();
        store.get_bytes().ok()
    }
}

impl<B: Backend> RecurrentPolicy for RecurrentCategoricalDistribution<B> {
    fn evaluate_sequence(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
        initial_state: Option<&Self::State>,
    ) -> anyhow::Result<RecurrentPolicyOutput<Self::Tensor, Self::State>> {
        ensure!(
            !observations.is_empty(),
            "recurrent sequence must not be empty"
        );
        ensure!(
            observations.len() == actions.len(),
            "recurrent observations and actions must have equal lengths"
        );
        let observations: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let (logits, final_state) = self.sequence_logits(observations, initial_state.cloned());
        let probs = softmax(logits.clone(), 1);
        let log_probs = log_softmax(logits, 1);
        let entropy = (probs * log_probs.clone())
            .neg()
            .sum_dim(1)
            .squeeze_dims(&[1]);
        let log_probs = (actions * log_probs).sum_dim(1).squeeze_dims(&[1]);
        Ok(RecurrentPolicyOutput {
            log_probs,
            entropy,
            final_state,
        })
    }

    fn std(&self) -> anyhow::Result<f32> {
        bail!("standard deviation is not defined for categorical distributions")
    }
}
