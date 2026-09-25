use std::f32;

use burn::module::ModuleDisplay;
use burn::module::{Module, Param};
use burn::tensor::cast::ToElement;
use burn::tensor::{Shape, TensorData};
use burn::{prelude::Backend, tensor::Tensor};
use burn_store::{ModuleStore, SafetensorsStore};
use r2l_core::{
    error::{Error, Result},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    rng::with_rng,
};
use rand_distr::{Distribution, StandardNormal};

use crate::networks::{Network, mlp::Mlp};

/// Diagonal-Gaussian Burn policy for Box action spaces.
///
/// This policy predicts the mean of a Gaussian action distribution and keeps a
/// learned diagonal log-standard-deviation parameter. It implements the
/// `r2l-core` [`Actor`] and [`Policy`] traits.
#[derive(Debug, Module)]
pub struct DiagGaussianDistribution<B: Backend, N: Module<B>> {
    mu_net: N,
    log_std: Param<Tensor<B, 2>>,
}

impl<B: Backend> DiagGaussianDistribution<B, Mlp<B>> {
    /// Builds a diagonal-Gaussian policy network.
    ///
    /// # Errors
    ///
    /// Returns an error if `mu_layers` is empty or the log-standard-deviation tensor cannot be
    /// created.
    pub fn build_mlp(
        mu_layers: &[usize],
        activation: ActivationFunction,
        log_std_init: f32,
    ) -> Result<Self> {
        let device = Default::default();
        let action_size = mu_layers
            .last()
            .copied()
            .ok_or_else(|| Error::invalid_parameter("mu_layers", "at least one layer", "[]"))?;
        let mu_net: Mlp<B> = Mlp::build(mu_layers, activation);
        let log_std = Param::from_data(
            TensorData::new(
                vec![log_std_init; action_size],
                Shape::new([1, action_size]),
            ),
            &device,
        );
        Ok(Self { mu_net, log_std })
    }
}

impl<B: Backend, N: Network<B>> DiagGaussianDistribution<B, N> {
    pub(crate) fn from_network(mu_net: N, action_size: usize, log_std_init: f32) -> Self {
        let log_std = Param::from_data(
            TensorData::new(
                vec![log_std_init; action_size],
                Shape::new([1, action_size]),
            ),
            &Default::default(),
        );
        Self { mu_net, log_std }
    }
}

impl<B: Backend, N: Network<B>> Actor for DiagGaussianDistribution<B, N> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device = Default::default();
        let mu = self.mu_net.forward(observation);
        let std = self.log_std.val().exp();
        let shape = mu.shape();
        let noise = with_rng(|rng| {
            (0..shape.num_elements())
                .map(|_| StandardNormal.sample(rng))
                .collect::<Vec<f32>>()
        });
        let noise = Tensor::from_data(TensorData::new(noise, shape), &device);
        let action = mu + noise * std;
        Ok(action.squeeze_dims(&[0]))
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        Ok(self.mu_net.forward(observation).squeeze_dims(&[0]))
    }
}

impl<B: Backend, N: Module<B> + ModuleDisplay> ToSafetensors for DiagGaussianDistribution<B, N> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).map_err(Error::wrap)?;
        store.get_bytes().map_err(Error::wrap)
    }
}

impl<B: Backend, N: Network<B>> Policy for DiagGaussianDistribution<B, N> {
    // FIXME: we probably want a differnt type states, actions etc. Alternatively we should have a
    // different trait, as log_probs are not really used during inference.
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        debug_assert!(!states.is_empty());
        debug_assert_eq!(states.len(), actions.len());
        let device = Default::default();
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let mu = self.mu_net.batch_forward(states);
        let log_std = self.log_std.val();
        let std = log_std.clone().exp();
        let var = std.clone() * std;
        let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
        let log_sqrt_2pi: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![log_sqrt_2pi; mu.shape().num_elements()], mu.shape()),
            &device,
        );
        let actions_minus_mu = actions - mu;
        let log_probs: Tensor<B, 2> = (actions_minus_mu.clone() * actions_minus_mu) / (2 * var);
        let log_probs = log_probs.neg() - log_std - log_sqrt_2pi;
        Ok(log_probs.sum_dim(1).squeeze())
    }

    fn entropy(&self, _states: &[Self::Tensor]) -> Result<Self::Tensor> {
        let device = Default::default();
        let log_std = self.log_std.val();
        let entropy_per_dim = log_std.clone()
            + Tensor::from_data(
                TensorData::new(
                    vec![
                        f32::midpoint((2. * f32::consts::PI).ln(), 1.);
                        log_std.shape().num_elements()
                    ],
                    log_std.shape(),
                ),
                &device,
            );
        Ok(entropy_per_dim.sum_dim(1).squeeze_dims(&[1]))
    }

    fn std(&self) -> Result<Option<f32>> {
        let std = self.log_std.val().exp().mean().into_scalar().to_f32();
        Ok(Some(std))
    }
}
