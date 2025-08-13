pub mod diagonal_distribution;

use burn::prelude::Backend;
use enum_dispatch::enum_dispatch;
use std::{f32, fmt::Debug};

use crate::{
    distributions::diagonal_distribution::DiagGaussianDistribution,
    env::{Action, Observation},
};

// TODO: I guess we should use Buffer here as well. Should this be a trait? I guess so
#[enum_dispatch]
// get action for a single observation
pub trait Distribution<O: Observation, A: Action>: Sync + Debug + 'static {
    fn get_action(&self, observation: O) -> (A, f32);

    // get logps for more than one states/action pairs
    fn log_probs(&self, states: &[O], actions: &[A]) -> Vec<f32>;

    // TODO: re evaluate this
    fn std(&self) -> f32;

    // TODO: entropy is also probably something
    fn entropy(&self) -> f32;

    fn resample_noise(&mut self) {}
}

// TODO: it looks like generics kill the enum dispatch
#[derive(Debug)]
pub enum DistribnutionKind<B: Backend> {
    Diagonal(DiagGaussianDistribution<B>),
}

impl<B: Backend, O: Observation, A: Action> Distribution<O, A> for DistribnutionKind<B> {
    fn get_action(&self, observation: O) -> (A, f32) {
        todo!()
    }

    fn log_probs(&self, states: &[O], actions: &[A]) -> Vec<f32> {
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
