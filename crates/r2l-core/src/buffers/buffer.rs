use std::any::TypeId;

use crate::{
    buffers::{
        Memory,
        reusable_vec::{ReusableVec, ReusableVecSlice},
    },
    tensor::R2lTensor,
};

pub enum TrajectoryTensorField<'a, T: R2lTensor> {
    Owned(Vec<T>),
    Borrowed(ReusableVecSlice<'a, T>),
}

impl<'a, T: R2lTensor> AsRef<[T]> for TrajectoryTensorField<'a, T> {
    fn as_ref(&self) -> &[T] {
        match self {
            Self::Owned(data) => data.as_slice(),
            Self::Borrowed(data) => data.as_ref(),
        }
    }
}

pub struct TrajectoryBatch<'a, T: R2lTensor> {
    states: TrajectoryTensorField<'a, T>,
    next_states: TrajectoryTensorField<'a, T>,
    actions: TrajectoryTensorField<'a, T>,
    rewards: ReusableVecSlice<'a, f32>,
    terminated: ReusableVecSlice<'a, bool>,
    truncated: ReusableVecSlice<'a, bool>,
}

impl<'a, T: R2lTensor> TrajectoryBatch<'a, T> {
    pub fn len(&self) -> usize {
        self.states.as_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn states(&self) -> &[T] {
        self.states.as_ref()
    }

    pub fn next_states(&self) -> &[T] {
        self.next_states.as_ref()
    }

    pub fn actions(&self) -> &[T] {
        self.actions.as_ref()
    }

    pub fn rewards(&self) -> &[f32] {
        self.rewards.as_ref()
    }

    pub fn terminated(&self) -> &[bool] {
        self.terminated.as_ref()
    }

    pub fn truncated(&self) -> &[bool] {
        self.truncated.as_ref()
    }
}

// the new buffer type? I guess we don't really need more
pub struct NewBuffer<T: R2lTensor> {
    states: ReusableVec<T>,
    next_states: ReusableVec<T>,
    actions: ReusableVec<T>,
    rewards: ReusableVec<f32>,
    terminated: ReusableVec<bool>,
    truncated: ReusableVec<bool>,
}

impl<T: R2lTensor> NewBuffer<T> {
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

    pub fn borrow_view(&mut self) -> TrajectoryBatch<'_, T> {
        let states = self.states.to_dropping_slice();
        let next_states = self.next_states.to_dropping_slice();
        let actions = self.actions.to_dropping_slice();
        TrajectoryBatch {
            states: TrajectoryTensorField::Borrowed(states),
            next_states: TrajectoryTensorField::Borrowed(next_states),
            actions: TrajectoryTensorField::Borrowed(actions),
            rewards: self.rewards.to_dropping_slice(),
            terminated: self.terminated.to_dropping_slice(),
            truncated: self.truncated.to_dropping_slice(),
        }
    }

    pub fn map_to_view<T2: From<T> + R2lTensor>(&mut self) -> TrajectoryBatch<'_, T2> {
        if TypeId::of::<T2>() == TypeId::of::<T>() {
            let buffer = self.borrow_view();
            let buffer = unsafe { std::mem::transmute(buffer) };
            return buffer;
        }
        let states = self.states.to_drain_iter().map(|t| t.into()).collect();
        let next_states = self.next_states.to_drain_iter().map(|t| t.into()).collect();
        let actions = self.actions.to_drain_iter().map(|t| t.into()).collect();
        TrajectoryBatch {
            states: TrajectoryTensorField::Owned(states),
            next_states: TrajectoryTensorField::Owned(next_states),
            actions: TrajectoryTensorField::Owned(actions),
            rewards: self.rewards.to_dropping_slice(),
            terminated: self.terminated.to_dropping_slice(),
            truncated: self.truncated.to_dropping_slice(),
        }
    }
}
