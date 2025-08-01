pub mod diagonal_distribution;

use burn::{prelude::Backend, tensor::Tensor};
use enum_dispatch::enum_dispatch;
use std::{f32, fmt::Debug};

use crate::distributions::diagonal_distribution::DiagGaussianDistribution;

#[enum_dispatch]
pub trait Distribution<B: Backend>: Sync + Debug + 'static {
    fn get_action(&self, observation: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>);
    fn log_probs(&self, states: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 1>;
    fn std(&self) -> f32;
    fn entropy(&self) -> Tensor<B, 1>;
    fn resample_noise(&mut self) {}
}

// TODO: it looks like generics kill the enum dispatch
#[derive(Debug)]
pub enum DistribnutionKind<B: Backend> {
    Diagonal(DiagGaussianDistribution<B>),
}

impl<B: Backend> Distribution<B> for DistribnutionKind<B> {
    fn get_action(&self, observation: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        todo!()
    }

    fn log_probs(&self, states: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 1> {
        todo!()
    }

    fn std(&self) -> f32 {
        todo!()
    }

    fn entropy(&self) -> Tensor<B, 1> {
        todo!()
    }

    fn resample_noise(&mut self) {
        todo!()
    }
}
