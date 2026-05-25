use crate::{buffers::buffer::TrajectoryView, tensor::R2lTensor};

pub trait TrajectoryBatchT<T: R2lTensor> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn states(&self) -> &[T];

    fn next_states(&self) -> &[T];

    fn actions(&self) -> &[T];

    fn rewards(&self) -> &[f32];

    fn terminated(&self) -> &[bool];

    fn truncated(&self) -> &[bool];
}

impl<'a, T: R2lTensor> TrajectoryBatchT<T> for TrajectoryView<'a, T> {
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
