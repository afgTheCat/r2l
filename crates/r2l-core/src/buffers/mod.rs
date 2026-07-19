use itertools::izip;

use crate::tensor::R2lTensor;

pub mod buffer;

/// One transition collected from an environment.
#[derive(Debug)]
pub struct Memory<T, S = ()> {
    /// Observation before the action.
    pub state: T,
    /// Observation after the action.
    pub next_state: T,
    /// Action selected by the actor.
    pub action: T,
    /// Recurrent actor state supplied when selecting the action.
    pub actor_state: Option<S>,
    /// Reward emitted by the environment.
    pub reward: f32,
    /// Whether the transition ended in a terminal state.
    pub terminated: bool,
    /// Whether the transition ended because of a time limit or external cutoff.
    pub truncated: bool,
}

impl<T, S> Memory<T, S> {
    /// Returns `true` when the transition ends the episode for any reason.
    pub fn is_done(&self) -> bool {
        self.terminated || self.truncated
    }
}

#[derive(Debug)]
pub struct MultiMemory<T: R2lTensor, S = ()> {
    pub last_states: Vec<T>,
    pub actions: Vec<T>,
    pub actor_states: Vec<Option<S>>,
    pub rewards: Vec<f32>,
    pub terminateds: Vec<bool>,
    pub truncateds: Vec<bool>,
}

impl<T: R2lTensor, S> MultiMemory<T, S> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            last_states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            actor_states: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            terminateds: Vec::with_capacity(capacity),
            truncateds: Vec::with_capacity(capacity),
        }
    }

    pub fn push_memory(&mut self, memory: Memory<T, S>) {
        let Memory {
            state,
            action,
            actor_state,
            reward,
            terminated,
            truncated,
            ..
        } = memory;
        self.last_states.push(state);
        self.actions.push(action);
        self.actor_states.push(actor_state);
        self.rewards.push(reward);
        self.terminateds.push(terminated);
        self.truncateds.push(truncated);
    }

    // TODO: maybe an iterator would be better
    pub fn into_memories(self, next_states: &[T]) -> Vec<Memory<T, S>> {
        let mut memories = Vec::with_capacity(self.last_states.len());
        let Self {
            last_states: states,
            actions,
            actor_states,
            rewards,
            terminateds,
            truncateds,
        } = self;
        for (state, next_state, action, actor_state, reward, terminated, truncated) in izip!(
            states,
            next_states,
            actions,
            actor_states,
            rewards,
            terminateds,
            truncateds
        ) {
            memories.push(Memory {
                state,
                next_state: next_state.clone(),
                action,
                actor_state,
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
    /// Recurrent state representation expected by the batch consumer.
    type State: Clone + Send + Sync + 'static;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn states(&self) -> &[T];

    fn next_states(&self) -> &[T];

    fn actions(&self) -> &[T];

    /// Incoming actor state aligned with each observation/action transition.
    fn actor_states(&self) -> &[Option<Self::State>];

    fn rewards(&self) -> &[f32];

    fn terminated(&self) -> &[bool];

    fn truncated(&self) -> &[bool];
}
