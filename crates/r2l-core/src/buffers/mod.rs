use crate::tensor::R2lTensor;

pub mod buffer;

/// One transition collected from an environment.
#[derive(Debug)]
pub struct Memory<T> {
    /// Observation before the action.
    pub state: T,
    /// Observation after the action.
    pub next_state: T,
    /// Action selected by the actor.
    pub action: T,
    /// Reward emitted by the environment.
    pub reward: f32,
    /// Whether the transition ended in a terminal state.
    pub terminated: bool,
    /// Whether the transition ended because of a time limit or external cutoff.
    pub truncated: bool,
}

impl<T> Memory<T> {
    /// Returns `true` when the transition ends the episode for any reason.
    pub fn is_done(&self) -> bool {
        self.terminated || self.truncated
    }
}

pub trait TrajectoryBatch<T: R2lTensor> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn states(&self) -> &[T];

    fn next_states(&self) -> &[T];

    fn actions(&self) -> &[T];

    fn rewards(&self) -> &[f32];

    fn terminated(&self) -> &[bool];

    fn truncated(&self) -> &[bool];
}
