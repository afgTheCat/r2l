use itertools::izip;

use crate::tensor::R2lTensor;

/// Owned and borrowed trajectory buffer types.
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

#[derive(Debug)]
/// A set of transitions collected from multiple environments in one step.
pub struct MultiMemory<T: R2lTensor> {
    last_states: Vec<T>,
    actions: Vec<T>,
    rewards: Vec<f32>,
    terminateds: Vec<bool>,
    truncateds: Vec<bool>,
}

impl<T: R2lTensor> MultiMemory<T> {
    /// Creates empty transition storage for up to `capacity` environments.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            last_states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            terminateds: Vec::with_capacity(capacity),
            truncateds: Vec::with_capacity(capacity),
        }
    }

    /// Adds one environment transition.
    pub fn push_memory(&mut self, memory: Memory<T>) {
        let Memory {
            state,
            action,
            reward,
            terminated,
            truncated,
            ..
        } = memory;
        self.last_states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.terminateds.push(terminated);
        self.truncateds.push(truncated);
    }

    /// Completes the stored transitions with their corresponding next states.
    ///
    /// Extra values on either side are ignored.
    pub fn into_memories(self, next_states: &[T]) -> Vec<Memory<T>> {
        let mut memories = Vec::with_capacity(self.last_states.len());
        let Self {
            last_states: states,
            actions,
            rewards,
            terminateds,
            truncateds,
        } = self;
        for (state, next_state, action, reward, terminated, truncated) in izip!(
            states,
            next_states,
            actions,
            rewards,
            terminateds,
            truncateds
        ) {
            memories.push(Memory {
                state,
                next_state: next_state.clone(),
                action,
                reward,
                terminated,
                truncated,
            });
        }
        memories
    }
}

/// Read-only access to a batch of aligned trajectory values.
pub trait TrajectoryBatch<T: R2lTensor> {
    /// Returns the number of transitions in the batch.
    fn len(&self) -> usize;

    /// Returns `true` when the batch contains no transitions.
    fn is_empty(&self) -> bool;

    /// Returns observations before each action.
    fn states(&self) -> &[T];

    /// Returns observations after each action.
    fn next_states(&self) -> &[T];

    /// Returns actions selected at each step.
    fn actions(&self) -> &[T];

    /// Returns rewards produced at each step.
    fn rewards(&self) -> &[f32];

    /// Returns terminal-state flags for each step.
    fn terminated(&self) -> &[bool];

    /// Returns truncation flags for each step.
    fn truncated(&self) -> &[bool];
}
