use ringbuffer::AllocRingBuffer;
use ringbuffer::RingBuffer;

use crate::{
    buffers::{ExpandableTrajectoryContainer, Memory, TrajectoryContainer},
    tensor::R2lTensor,
};

/// Ring-buffer trajectory storage with a fixed capacity.
///
/// When more than `len` transitions are pushed, older transitions are evicted
/// by the underlying ring buffers.
pub struct FixedSizeStateBuffer<T: R2lTensor> {
    /// Maximum number of transitions retained.
    pub len: usize,
    /// Observations before each action.
    pub states: AllocRingBuffer<T>,
    /// Observations after each action.
    pub next_states: AllocRingBuffer<T>,
    /// Rewards for each transition.
    pub rewards: AllocRingBuffer<f32>,
    /// Actions for each transition.
    pub action: AllocRingBuffer<T>,
    /// Terminal-state flags.
    pub terminated: AllocRingBuffer<bool>,
    /// Time-limit or external-cutoff flags.
    pub truncated: AllocRingBuffer<bool>,
}

impl<T: R2lTensor> FixedSizeStateBuffer<T> {
    /// Creates an empty fixed-capacity trajectory buffer.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            states: AllocRingBuffer::new(len),
            next_states: AllocRingBuffer::new(len),
            rewards: AllocRingBuffer::new(len),
            action: AllocRingBuffer::new(len),
            terminated: AllocRingBuffer::new(len),
            truncated: AllocRingBuffer::new(len),
        }
    }
}

impl<T: R2lTensor> TrajectoryContainer for FixedSizeStateBuffer<T> {
    type Tensor = T;

    fn len(&self) -> usize {
        self.len
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

impl<T: R2lTensor> ExpandableTrajectoryContainer for FixedSizeStateBuffer<T> {
    fn push(&mut self, memory: Memory<Self::Tensor>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        } = memory;
        self.states.enqueue(state);
        self.next_states.enqueue(next_state);
        self.action.enqueue(action);
        self.rewards.enqueue(reward);
        self.terminated.enqueue(terminated);
        self.truncated.enqueue(truncated);
    }
}
