use crate::buffers::{ContainerView, Memory, TrajectoryContainer2};
use crate::{tensor::R2lTensor, utils::bring_buffer::BringBuffer};

pub struct FixedSizeStateBuffer2<T: R2lTensor> {
    /// Maximum number of transitions retained.
    pub len: usize,
    /// Observations before each action.
    pub states: BringBuffer<T>,
    /// Observations after each action.
    pub next_states: BringBuffer<T>,
    /// Rewards for each transition.
    pub rewards: BringBuffer<f32>,
    /// Actions for each transition.
    pub action: BringBuffer<T>,
    /// Terminal-state flags.
    pub terminated: BringBuffer<bool>,
    /// Time-limit or external-cutoff flags.
    pub truncated: BringBuffer<bool>,
}

impl<T: R2lTensor> FixedSizeStateBuffer2<T> {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            states: BringBuffer::new(len),
            next_states: BringBuffer::new(len),
            rewards: BringBuffer::new(len),
            action: BringBuffer::new(len),
            terminated: BringBuffer::new(len),
            truncated: BringBuffer::new(len),
        }
    }

    fn rollout_complete(&self) -> bool {
        self.terminated
            .read_last()
            .zip(self.truncated.read_last())
            .map(|(terminated, truncated)| *terminated || *truncated)
            .unwrap_or(false)
    }

    fn drain_buffer<U>(buffer: &mut BringBuffer<U>) -> Vec<U> {
        let mut values = Vec::with_capacity(buffer.len());
        while let Some(value) = buffer.pop_back() {
            values.push(value);
        }
        values.reverse();
        values
    }
}

impl<T: R2lTensor> TrajectoryContainer2 for FixedSizeStateBuffer2<T> {
    type Tensor = T;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn states(&self) -> Option<&[Self::Tensor]> {
        self.rollout_complete()
            .then(|| self.states.try_slice())
            .flatten()
    }

    fn next_states(&self) -> Option<&[Self::Tensor]> {
        self.rollout_complete()
            .then(|| self.next_states.try_slice())
            .flatten()
    }

    fn actions(&self) -> Option<&[Self::Tensor]> {
        self.rollout_complete()
            .then(|| self.action.try_slice())
            .flatten()
    }

    fn rewards(&self) -> Option<&[f32]> {
        self.rollout_complete()
            .then(|| self.rewards.try_slice())
            .flatten()
    }

    fn terminated(&self) -> Option<&[bool]> {
        self.rollout_complete()
            .then(|| self.terminated.try_slice())
            .flatten()
    }

    fn truncated(&self) -> Option<&[bool]> {
        self.rollout_complete()
            .then(|| self.truncated.try_slice())
            .flatten()
    }

    fn begin_rollout(&mut self) {
        // we don't need to clear the buffer
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
        self.states.enqueu(state);
        self.next_states.enqueu(next_state);
        self.action.enqueu(action);
        self.rewards.enqueu(reward);
        self.terminated.enqueu(terminated);
        self.truncated.enqueu(truncated);
    }

    fn pop(&mut self) -> Option<Memory<Self::Tensor>> {
        Some(Memory {
            state: self.states.pop_back()?,
            next_state: self.next_states.pop_back()?,
            action: self.action.pop_back()?,
            reward: self.rewards.pop_back()?,
            terminated: self.terminated.pop_back()?,
            truncated: self.truncated.pop_back()?,
        })
    }

    fn container_view(&mut self) -> ContainerView<Self::Tensor> {
        ContainerView {
            states: Self::drain_buffer(&mut self.states),
            next_states: Self::drain_buffer(&mut self.next_states),
            actions: Self::drain_buffer(&mut self.action),
            rewards: Self::drain_buffer(&mut self.rewards),
            terminated: Self::drain_buffer(&mut self.terminated),
            trancuated: Self::drain_buffer(&mut self.truncated),
        }
    }
}
