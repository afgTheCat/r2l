use itertools::izip;

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

#[derive(Debug)]
pub struct MultiMemory<T> {
    pub states: Vec<T>,
    pub next_states: Vec<T>,
    pub actions: Vec<T>,
    pub rewards: Vec<f32>,
    pub terminateds: Vec<bool>,
    pub truncateds: Vec<bool>,
}

impl<T> MultiMemory<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            terminateds: Vec::with_capacity(capacity),
            truncateds: Vec::with_capacity(capacity),
        }
    }

    pub fn push_memory(&mut self, memory: Memory<T>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        } = memory;
        self.states.push(state);
        self.next_states.push(next_state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.terminateds.push(terminated);
        self.truncateds.push(truncated);
    }

    // TODO: maybe an iterator would be better
    pub fn into_memories(self) -> Vec<Memory<T>> {
        let mut memories = Vec::with_capacity(self.states.len());
        let Self {
            states,
            next_states,
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
                next_state,
                action,
                reward,
                terminated,
                truncated,
            });
        }
        memories
    }
}

// TODO: do we need this trait?
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
