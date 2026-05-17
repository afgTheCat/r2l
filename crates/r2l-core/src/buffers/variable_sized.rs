use crate::{
    buffers::{ExpandableTrajectoryContainer, Memory, TrajectoryContainer, TrajectoryContainer2},
    tensor::R2lTensor,
};

/// Growable trajectory storage backed by vectors.
#[derive(Clone)]
pub struct VariableSizedStateBuffer<T: R2lTensor> {
    /// Observations before each action.
    pub states: Vec<T>,
    /// Observations after each action.
    pub next_states: Vec<T>,
    /// Rewards for each transition.
    pub rewards: Vec<f32>,
    /// Actions for each transition.
    pub action: Vec<T>,
    /// Terminal-state flags.
    pub terminated: Vec<bool>,
    /// Time-limit or external-cutoff flags.
    pub truncated: Vec<bool>,
}

impl<T: R2lTensor> Default for VariableSizedStateBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: R2lTensor> VariableSizedStateBuffer<T> {
    /// Creates an empty growable trajectory buffer.
    pub fn new() -> Self {
        Self {
            states: vec![],
            next_states: vec![],
            rewards: vec![],
            action: vec![],
            terminated: vec![],
            truncated: vec![],
        }
    }

    fn rollout_complete(&self) -> bool {
        self.terminated
            .last()
            .zip(self.truncated.last())
            .map(|(terminated, truncated)| *terminated || *truncated)
            .unwrap_or(false)
    }
}

impl<T: R2lTensor> TrajectoryContainer for VariableSizedStateBuffer<T> {
    type Tensor = T;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn states(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.states.iter()
    }

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.next_states.iter()
    }

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.action.iter()
    }

    fn rewards(&self) -> impl Iterator<Item = f32> {
        self.rewards.iter().copied()
    }

    fn terminated(&self) -> impl Iterator<Item = bool> {
        self.terminated.iter().copied()
    }

    fn truncated(&self) -> impl Iterator<Item = bool> {
        self.truncated.iter().copied()
    }
}

impl<T: R2lTensor> ExpandableTrajectoryContainer for VariableSizedStateBuffer<T> {
    fn begin_rollout(&mut self) {
        self.states.clear();
        self.next_states.clear();
        self.action.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
    }

    fn push(&mut self, memory: Memory<Self::Tensor>) {
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
        self.action.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.truncated.push(truncated);
    }
}

impl<T: R2lTensor> TrajectoryContainer2 for VariableSizedStateBuffer<T> {
    type Tensor = T;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn states(&self) -> Option<&[Self::Tensor]> {
        self.rollout_complete().then_some(self.states.as_slice())
    }

    fn next_states(&self) -> Option<&[Self::Tensor]> {
        self.rollout_complete()
            .then_some(self.next_states.as_slice())
    }

    fn actions(&self) -> Option<&[Self::Tensor]> {
        self.rollout_complete().then_some(self.action.as_slice())
    }

    fn rewards(&self) -> Option<&[f32]> {
        self.rollout_complete().then_some(self.rewards.as_slice())
    }

    fn terminated(&self) -> Option<&[bool]> {
        self.rollout_complete()
            .then_some(self.terminated.as_slice())
    }

    fn truncated(&self) -> &[bool] {
        self.truncated.as_slice()
    }

    fn begin_rollout(&mut self) {
        self.states.clear();
        self.next_states.clear();
        self.action.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
    }

    fn push(&mut self, memory: Memory<Self::Tensor>) {
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
        self.action.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.truncated.push(truncated);
    }

    fn pop(&mut self) -> Option<Memory<Self::Tensor>> {
        Some(Memory {
            state: self.states.pop()?,
            next_state: self.next_states.pop()?,
            action: self.action.pop()?,
            reward: self.rewards.pop()?,
            terminated: self.terminated.pop()?,
            truncated: self.truncated.pop()?,
        })
    }
}
