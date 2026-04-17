use std::f32;

use anyhow::Result;
use burn::module::{Module, Param};
use burn::tensor::cast::ToElement;
use burn::tensor::{Distribution as BurnDistribution, Shape, TensorData};
use burn::{prelude::Backend, tensor::Tensor as BurnTensor};
use r2l_core::models::{Actor, Policy};

use crate::sequential::Sequential;

#[derive(Debug, Module)]
pub struct DiagGaussianDistribution<B: Backend> {
    pub mu_net: Sequential<B>,
    pub log_std: Param<BurnTensor<B, 2>>,
}

impl<B: Backend> DiagGaussianDistribution<B> {
    pub fn new(mu_net: Sequential<B>, log_std: Param<BurnTensor<B, 2>>) -> Self {
        Self { mu_net, log_std }
    }

    pub fn build(mu_layers: &[usize]) -> Self {
        let device = Default::default();
        let action_size = *mu_layers.last().unwrap();
        let mu_net: Sequential<B> = Sequential::build(mu_layers);
        let log_std = Param::from_data(
            TensorData::zeros::<f32, _>(Shape::new([1, action_size])),
            &device,
        );
        Self { mu_net, log_std }
    }
}

impl<B: Backend> Actor for DiagGaussianDistribution<B> {
    type Tensor = BurnTensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device: <B as Backend>::Device = Default::default();
        let observation: BurnTensor<B, 2> = observation.unsqueeze();
        let mu = self.mu_net.forward(observation);
        let std = self.log_std.val().exp();
        let noise = BurnTensor::random(mu.shape(), BurnDistribution::Normal(0., 1.), &device);
        let action = mu + noise * std;
        Ok(action.squeeze_dims(&[1]))
    }
}

impl<B: Backend> Policy for DiagGaussianDistribution<B> {
    // FIXME: we probably want a differnt type states, actions etc. Alternatively we should have a
    // different trait, as log_probs are not really used during inference.
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        let device: <B as Backend>::Device = Default::default();
        let states: BurnTensor<B, 2> = BurnTensor::stack(states.to_vec(), 0);
        let actions: BurnTensor<B, 2> = BurnTensor::stack(actions.to_vec(), 0);
        let mu = self.mu_net.forward(states);
        let log_std = self.log_std.val();
        let std = log_std.clone().exp();
        let var = std.clone() * std;
        let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
        let log_sqrt_2pi: BurnTensor<B, 2> = BurnTensor::from_data(
            TensorData::new(
                vec![log_sqrt_2pi; mu.shape().num_elements()],
                mu.shape().dims,
            ),
            &device,
        );
        let actions_minus_mu = actions - mu;
        let log_probs: BurnTensor<B, 2> = (actions_minus_mu.clone() * actions_minus_mu) / (2 * var);
        let log_probs = log_probs.neg() - log_std - log_sqrt_2pi;
        Ok(log_probs.sum_dim(1).squeeze())
    }

    fn entropy(&self, _states: &[Self::Tensor]) -> Result<Self::Tensor> {
        let device: <B as Backend>::Device = Default::default();
        let log_std = self.log_std.val();
        let entropy_per_dim = log_std.clone()
            + BurnTensor::from_data(
                TensorData::new(
                    vec![0.5 * ((2. * f32::consts::PI).ln() + 1.); log_std.shape().num_elements()],
                    log_std.shape().dims,
                ),
                &device,
            );
        Ok(entropy_per_dim.sum_dim(1).squeeze_dims(&[1]))
    }

    fn std(&self) -> Result<f32> {
        let std = self.log_std.val().exp().mean().into_scalar().to_f32();
        Ok(std)
    }

    fn resample_noise(&mut self) -> Result<()> {
        todo!()
    }
}
