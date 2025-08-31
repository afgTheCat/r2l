use crate::sequential::Sequential;
use anyhow::Result;
use burn::{prelude::Backend, tensor::Tensor};
use r2l_core::distributions::Distribution;

pub struct DiagGaussianDistribution<B: Backend> {
    pub mu_net: Sequential<B>,
    pub value_net: Sequential<B>,
}

impl<B: Backend> DiagGaussianDistribution<B> {
    pub fn new(mu_net: Sequential<B>, value_net: Sequential<B>) -> Self {
        Self { mu_net, value_net }
    }
}

// impl<B> Distribution for DiagGaussianDistribution<B> {
//     type Tensor = Tensor<B, 1>;
//
//     fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn log_probs(&self, states: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn std(&self) -> Result<f32> {
//         todo!()
//     }
//
//     fn resample_noise(&mut self) -> Result<()> {
//         todo!()
//     }
//
//     fn entropy(&self) -> Result<Self::Tensor> {
//         todo!()
//     }
// }
