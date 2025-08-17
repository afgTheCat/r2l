use crate::env::{Action, Logp, Observation};
use std::{f32, fmt::Debug};

pub trait Distribution<O: Observation, A: Action, L: Logp>: Sync + Debug + 'static {
    // get action for a single observation
    fn get_action(&self, observation: O) -> (A, f32);

    // get logps for more than one states/action pairs
    fn log_probs(&self, states: &[O], actions: &[A]) -> Vec<L>;

    // TODO: re evaluate this
    fn std(&self) -> f32;

    // TODO: entropy is also probably something
    fn entropy(&self) -> f32;

    fn resample_noise(&mut self) {}
}
