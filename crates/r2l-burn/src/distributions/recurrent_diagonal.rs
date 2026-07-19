use std::f32;

use anyhow::ensure;
use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig, Rnn, RnnConfig, RnnState},
    prelude::Backend,
    tensor::{Distribution, Shape, Tensor, TensorData, cast::ToElement},
};
use burn_store::{ModuleStore, SafetensorsStore};
use r2l_core::models::{ActivationFunction, Actor, RecurrentPolicy, RecurrentPolicyOutput};

use crate::sequential::Sequential;

/// Recurrent diagonal-Gaussian policy for continuous action spaces.
#[derive(Debug, Module)]
pub struct RecurrentDiagGaussianDistribution<B: Backend> {
    encoder: Sequential<B>,
    recurrent: Rnn<B>,
    mu: Linear<B>,
    log_std: Param<Tensor<B, 2>>,
}

impl<B: Backend> RecurrentDiagGaussianDistribution<B> {
    /// Builds a recurrent diagonal-Gaussian policy.
    ///
    /// `layers` contains the observation size, optional encoder sizes, and the
    /// final action size. The last encoder size is also the recurrent size.
    pub fn build(layers: &[usize]) -> Self {
        assert!(
            layers.len() >= 2,
            "recurrent diagonal policy requires input and output sizes"
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
        let mu = LinearConfig::new(recurrent_size, action_size)
            .with_bias(true)
            .init::<B>(&device);
        let log_std = Param::from_data(
            TensorData::zeros::<f32, _>(Shape::new([1, action_size])),
            &device,
        );
        Self {
            encoder,
            recurrent,
            mu,
            log_std,
        }
    }

    fn sequence_mu(
        &self,
        observations: Tensor<B, 2>,
        state: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let encoded = self.encoder.forward(observations).unsqueeze_dim(0);
        let state = state.map(RnnState::new);
        let (output, state) = self.recurrent.forward(encoded, state);
        let output = output.squeeze_dim(0);
        (self.mu.forward(output), state.hidden)
    }
}

impl<B: Backend> Actor for RecurrentDiagGaussianDistribution<B> {
    type Tensor = Tensor<B, 1>;
    type State = Tensor<B, 2>;

    fn action(
        &self,
        observation: Self::Tensor,
        state: Option<Self::State>,
    ) -> anyhow::Result<(Self::Tensor, Self::State)> {
        let device = Default::default();
        let (mu, state) = self.sequence_mu(observation.unsqueeze(), state);
        let std = self.log_std.val().exp();
        let noise = Tensor::random(mu.shape(), Distribution::Normal(0., 1.), &device);
        Ok(((mu + noise * std).squeeze_dims(&[0]), state))
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).unwrap();
        store.get_bytes().ok()
    }
}

impl<B: Backend> RecurrentPolicy for RecurrentDiagGaussianDistribution<B> {
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
        let device = Default::default();
        let observations: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let (mu, final_state) = self.sequence_mu(observations, initial_state.cloned());
        let [steps, action_size] = mu.dims();
        let log_std = self.log_std.val().expand([steps, action_size]);
        let std = log_std.clone().exp();
        let variance = std.clone() * std;
        let log_sqrt_2pi = 0.5 * (2. * f32::consts::PI).ln();
        let log_sqrt_2pi = Tensor::from_data(
            TensorData::new(
                vec![log_sqrt_2pi; steps * action_size],
                [steps, action_size],
            ),
            &device,
        );
        let difference = actions - mu;
        let log_probs = (difference.clone() * difference) / (variance * 2.);
        let log_probs = log_probs.neg() - log_std.clone() - log_sqrt_2pi;
        let log_probs = log_probs.sum_dim(1).squeeze_dims(&[1]);
        let entropy_constant = 0.5 * ((2. * f32::consts::PI).ln() + 1.);
        let entropy_constant = Tensor::from_data(
            TensorData::new(
                vec![entropy_constant; steps * action_size],
                [steps, action_size],
            ),
            &device,
        );
        let entropy = (log_std + entropy_constant).sum_dim(1).squeeze_dims(&[1]);
        Ok(RecurrentPolicyOutput {
            log_probs,
            entropy,
            final_state,
        })
    }

    fn std(&self) -> anyhow::Result<f32> {
        Ok(self.log_std.val().exp().mean().into_scalar().to_f32())
    }
}
