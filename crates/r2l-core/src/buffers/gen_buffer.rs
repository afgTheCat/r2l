use crate::{
    buffers::buffer::{Buffer, TrajectoryBatch},
    tensor::R2lTensor,
};

trait TrajectoryBatchT<T: R2lTensor> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn states(&self) -> &[T];

    fn next_states(&self) -> &[T];

    fn actions(&self) -> &[T];

    fn rewards(&self) -> &[f32];

    fn terminated(&self) -> &[bool];

    fn truncated(&self) -> &[bool];
}

impl<'a, T: R2lTensor> TrajectoryBatchT<T> for TrajectoryBatch<'a, T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn states(&self) -> &[T] {
        self.states()
    }

    fn next_states(&self) -> &[T] {
        self.next_states()
    }

    fn actions(&self) -> &[T] {
        self.actions()
    }

    fn rewards(&self) -> &[f32] {
        self.rewards()
    }

    fn terminated(&self) -> &[bool] {
        self.terminated()
    }

    fn truncated(&self) -> &[bool] {
        self.truncated()
    }
}

trait BufferT {
    type Tensor: R2lTensor;

    fn map_to_view<T2: R2lTensor + From<Self::Tensor>>(&mut self) -> impl TrajectoryBatchT<T2>;
}

impl<T: R2lTensor> BufferT for Buffer<T> {
    type Tensor = T;

    fn map_to_view<T2: R2lTensor + From<Self::Tensor>>(&mut self) -> impl TrajectoryBatchT<T2> {
        self.map_to_view()
    }
}
