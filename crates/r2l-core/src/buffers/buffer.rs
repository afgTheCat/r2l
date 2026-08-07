use crate::{
    buffers::{Memory, TrajectoryBatch},
    tensor::R2lTensor,
};

/// Owned, structure-of-arrays storage for one environment trajectory.
#[derive(Clone)]
pub struct TrajectoryBuffer<T: R2lTensor> {
    states: Vec<T>,
    next_states: Vec<T>,
    actions: Vec<T>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl<T: R2lTensor> Default for TrajectoryBuffer<T> {
    fn default() -> Self {
        Self {
            states: Default::default(),
            next_states: Default::default(),
            actions: Default::default(),
            rewards: Default::default(),
            terminated: Default::default(),
            truncated: Default::default(),
        }
    }
}

impl<T: R2lTensor> TrajectoryBuffer<T> {
    /// Removes all stored transitions while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.states.clear();
        self.next_states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
    }

    /// Appends one transition to the buffer.
    pub fn push(&mut self, memory: Memory<T>) {
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
        self.terminated.push(terminated);
        self.truncated.push(truncated);
    }

    /// Replaces the most recently stored next state, if one exists.
    pub fn replace_last_next_state(&mut self, next_state: T) {
        if let Some(last_next_state) = self.next_states.last_mut() {
            *last_next_state = next_state;
        }
    }

    /// Returns the number of stored transitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` when the buffer contains no transitions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns terminal-state flags.
    #[must_use]
    pub fn terminated(&self) -> &[bool] {
        &self.terminated
    }

    /// Returns truncation flags.
    #[must_use]
    pub fn truncated(&self) -> &[bool] {
        &self.truncated
    }

    /// Returns stored rewards.
    #[must_use]
    pub fn rewards(&self) -> &[f32] {
        &self.rewards
    }

    /// Returns mutable access to the stored rewards.
    pub fn rewards_mut(&mut self) -> &mut [f32] {
        &mut self.rewards
    }

    /// Borrows the buffer as aligned trajectory slices.
    #[must_use]
    pub fn to_trajectory_view(&self) -> TrajectoryView<'_, T> {
        TrajectoryView {
            states: &self.states,
            next_states: &self.next_states,
            actions: &self.actions,
            rewards: &self.rewards,
            terminated: &self.terminated,
            truncated: &self.truncated,
        }
    }
}

/// Borrowed view over the aligned fields of a [`TrajectoryBuffer`].
pub struct TrajectoryView<'a, T: R2lTensor> {
    /// Observations before each action.
    pub states: &'a [T],
    /// Observations after each action.
    pub next_states: &'a [T],
    /// Actions selected at each step.
    pub actions: &'a [T],
    /// Rewards produced at each step.
    pub rewards: &'a [f32],
    /// Terminal-state flags for each step.
    pub terminated: &'a [bool],
    /// Truncation flags for each step.
    pub truncated: &'a [bool],
}

impl<T: R2lTensor> TrajectoryBatch<T> for TrajectoryView<'_, T> {
    fn len(&self) -> usize {
        self.states.len()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn states(&self) -> &[T] {
        self.states
    }

    fn next_states(&self) -> &[T] {
        self.next_states
    }

    fn actions(&self) -> &[T] {
        self.actions
    }

    fn rewards(&self) -> &[f32] {
        self.rewards
    }

    fn terminated(&self) -> &[bool] {
        self.terminated
    }

    fn truncated(&self) -> &[bool] {
        self.truncated
    }
}

impl<T: R2lTensor> TrajectoryView<'_, T> {
    /// Iterates over combined termination and truncation flags.
    pub fn dones(&self) -> impl Iterator<Item = bool> {
        self.terminated
            .iter()
            .zip(self.truncated.iter())
            .map(|(terminated, truncated)| *terminated || *truncated)
    }

    /// Counts transitions that end an episode.
    #[must_use]
    pub fn episode_terminations(&self) -> usize {
        self.dones().filter(|x| *x).count()
    }
}
