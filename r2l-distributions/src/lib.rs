use crate::diagonal_distribution::DiagGaussianDistribution;
use burn::prelude::Backend;
use r2l_core2::{
    distributions::Distribution,
    env::{Action, Logp, Observation},
};
mod diagonal_distribution;

// TODO: it looks like generics kill the enum dispatch
#[derive(Debug, Clone)]
pub enum DistribnutionKind<B: Backend> {
    Diagonal(DiagGaussianDistribution<B>),
}

impl<B: Backend, O: Observation, A: Action, L: Logp> Distribution<O, A, L>
    for DistribnutionKind<B>
{
    fn get_action(&self, observation: O) -> (A, f32) {
        todo!()
    }

    fn log_probs(&self, states: &[O], actions: &[A]) -> Vec<L> {
        todo!()
    }

    fn std(&self) -> f32 {
        todo!()
    }

    fn entropy(&self) -> f32 {
        todo!()
    }

    fn resample_noise(&mut self) {
        todo!()
    }
}
