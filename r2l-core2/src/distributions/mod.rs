use crate::env::{Action, Observation};
use std::{f32, fmt::Debug};

pub trait Distribution<O: Observation, A: Action>: Sync + Debug + 'static {
    // get action for a single observation
    fn get_action(&self, observation: O) -> (A, f32);

    // get logps for more than one states/action pairs
    fn log_probs(&self, states: &[O], actions: &[A]) -> Vec<f32>;

    // TODO: re evaluate this
    fn std(&self) -> f32;

    // TODO: entropy is also probably something
    fn entropy(&self) -> f32;

    fn resample_noise(&mut self) {}
}
