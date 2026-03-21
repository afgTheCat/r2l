use ringbuffer::AllocRingBuffer;

use crate::{
    sampler5::buffer::{ExpandableTrajectoryContainer, Memory, TrajectoryContainer},
    tensor::R2lTensor,
};
use ringbuffer::RingBuffer;

pub struct FixedSizeStateBuffer<T: R2lTensor> {
    pub len: usize,
    pub states: AllocRingBuffer<T>,
    pub next_states: AllocRingBuffer<T>,
    pub rewards: AllocRingBuffer<f32>,
    pub action: AllocRingBuffer<T>,
    pub terminated: AllocRingBuffer<bool>,
    pub trancuated: AllocRingBuffer<bool>,
}

impl<T: R2lTensor> FixedSizeStateBuffer<T> {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            states: AllocRingBuffer::new(len),
            next_states: AllocRingBuffer::new(len),
            rewards: AllocRingBuffer::new(len),
            action: AllocRingBuffer::new(len),
            terminated: AllocRingBuffer::new(len),
            trancuated: AllocRingBuffer::new(len),
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

    fn trancuated(&self) -> impl Iterator<Item = bool> {
        self.trancuated.iter().copied()
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
            trancuated,
        } = memory;
        self.states.enqueue(state);
        self.next_states.enqueue(next_state);
        self.action.enqueue(action);
        self.rewards.enqueue(reward);
        self.terminated.enqueue(terminated);
        self.trancuated.enqueue(trancuated);
    }
}
