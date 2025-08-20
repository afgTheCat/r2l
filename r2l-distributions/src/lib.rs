use crate::diagonal_distribution::DiagGaussianDistribution;
use burn::{prelude::Backend, tensor::Tensor};
use r2l_core2::distributions::Distribution;
mod diagonal_distribution;

// TODO: it looks like generics kill the enum dispatch
#[derive(Debug, Clone)]
pub enum DistribnutionKind<B: Backend> {
    Diagonal(DiagGaussianDistribution<B>),
}

impl<B: Backend> Distribution for DistribnutionKind<B> {
    type Observation = Tensor<B, 2>;
    type Action = Tensor<B, 2>;
    type Logp = Tensor<B, 2>;

    fn get_action(&self, observation: Self::Observation) -> (Self::Action, f32) {
        todo!()
    }

    fn log_probs(&self, states: &[Self::Observation], actions: &[Self::Action]) -> Vec<Self::Logp> {
        todo!()
    }

    fn entropy(&self) -> f32 {
        todo!()
    }

    fn std(&self) -> f32 {
        todo!()
    }

    fn resample_noise(&mut self) {
        todo!()
    }
}
