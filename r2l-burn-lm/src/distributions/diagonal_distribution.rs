use crate::sequential::Sequential;
use anyhow::Result;
use burn::module::Module;
use burn::tensor::cast::ToElement;
use burn::tensor::{Distribution as BurnDistribution, Shape};
use burn::{prelude::Backend, tensor::Tensor};
use r2l_core::distributions::Distribution;

#[derive(Debug, Module)]
pub struct DiagGaussianDistribution<B: Backend> {
    pub mu_net: Sequential<B>,
    pub value_net: Sequential<B>,
    pub log_std: Tensor<B, 2>,
}

impl<B: Backend> DiagGaussianDistribution<B> {
    pub fn new(mu_net: Sequential<B>, value_net: Sequential<B>, log_std: Tensor<B, 2>) -> Self {
        Self {
            mu_net,
            value_net,
            log_std,
        }
    }

    pub fn build(mu_layers: &[usize], value_layers: &[usize]) -> Self {
        let device = Default::default();
        let action_size = *mu_layers.last().unwrap();
        let mu_net: Sequential<B> = Sequential::build(mu_layers);
        let value_net: Sequential<B> = Sequential::build(value_layers);
        let log_std: Tensor<B, 2> = Tensor::random(
            Shape::new([action_size]),
            burn::tensor::Distribution::Normal(0., 1.),
            &device,
        );
        Self {
            mu_net,
            log_std,
            value_net,
        }
    }
}

impl<B: Backend> Distribution for DiagGaussianDistribution<B> {
    type Tensor = Tensor<B, 1>;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let device: <B as Backend>::Device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let mu = self.mu_net.forward(observation);
        let std = self.log_std.clone().exp();
        let noise = Tensor::random(mu.shape(), BurnDistribution::Normal(0., 1.), &device);
        let action = mu + noise * std;
        Ok(action.squeeze(0))
    }

    // FIXME: we probably want a differnt type states, actions etc. Alternatively we should have a
    // different trait, as log_probs are not really used during inference.
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        todo!()
    }

    fn entropy(&self) -> Result<Self::Tensor> {
        todo!()
    }

    fn std(&self) -> Result<f32> {
        let std = self.log_std.clone().exp().mean().into_scalar().to_f32();
        Ok(std)
    }

    fn resample_noise(&mut self) -> Result<()> {
        todo!()
    }
}
