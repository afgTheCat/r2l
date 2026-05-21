use crate::{
    buffers::reusable_vec::{ReusableVec, ReusableVecSlice},
    tensor::R2lTensor,
};

enum TrajectoryTensorField<'a, T: R2lTensor> {
    Owned(Vec<T>),
    Borrowed(ReusableVecSlice<'a, T>),
}

struct TrajecectoryBatch<'a, T: R2lTensor> {
    states: TrajectoryTensorField<'a, T>,
    next_states: TrajectoryTensorField<'a, T>,
    actions: TrajectoryTensorField<'a, T>,
    rewards: ReusableVecSlice<'a, f32>,
    terminated: ReusableVecSlice<'a, bool>,
    truncated: ReusableVecSlice<'a, bool>,
}

// the new buffer type? I guess we don't really need more
struct Buffer<T: R2lTensor> {
    states: ReusableVec<T>,
    next_states: ReusableVec<T>,
    actions: ReusableVec<T>,
    rewards: ReusableVec<f32>,
    terminated: ReusableVec<bool>,
    truncated: ReusableVec<bool>,
}

impl<T: R2lTensor> Buffer<T> {
    pub fn map_to_view<T2: From<T> + R2lTensor>(&mut self) -> TrajecectoryBatch<T2> {
        let states = self.states.to_drain_iter().map(|t| t.into()).collect();
        let next_states = self.next_states.to_drain_iter().map(|t| t.into()).collect();
        let actions = self.actions.to_drain_iter().map(|t| t.into()).collect();
        TrajecectoryBatch {
            states: TrajectoryTensorField::Owned(states),
            next_states: TrajectoryTensorField::Owned(next_states),
            actions: TrajectoryTensorField::Owned(actions),
            rewards: self.rewards.to_dropping_slice(),
            terminated: self.terminated.to_dropping_slice(),
            truncated: self.truncated.to_dropping_slice(),
        }
    }

    pub fn borrow_view(&mut self) -> TrajecectoryBatch<T> {
        let states = self.states.to_dropping_slice();
        let next_states = self.next_states.to_dropping_slice();
        let actions = self.actions.to_dropping_slice();
        TrajecectoryBatch {
            states: TrajectoryTensorField::Borrowed(states),
            next_states: TrajectoryTensorField::Borrowed(next_states),
            actions: TrajectoryTensorField::Borrowed(actions),
            rewards: self.rewards.to_dropping_slice(),
            terminated: self.terminated.to_dropping_slice(),
            truncated: self.truncated.to_dropping_slice(),
        }
    }
}
