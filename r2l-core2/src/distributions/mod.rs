use crate::env::{Action, Logp, Observation};
use std::{f32, fmt::Debug};

pub trait Distribution: Sync + Debug + 'static {
    type Observation: Observation;
    type Action: Action;
    type Logp: Logp;

    // get action for a single observation
    fn get_action(&self, observation: Self::Observation) -> (Self::Action, f32);

    // get logps for more than one states/action pairs
    fn log_probs(&self, states: &[Self::Observation], actions: &[Self::Action]) -> Vec<Self::Logp>;

    // TODO: re evaluate this
    fn std(&self) -> f32;

    // TODO: entropy is also probably something
    fn entropy(&self) -> f32;

    fn resample_noise(&mut self) {}
}
