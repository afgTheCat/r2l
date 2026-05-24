use std::any::TypeId;

use crate::{
    buffers::{buffer::TrajectoryView, gen_buffer::TrajectoryBatchT},
    tensor::R2lTensor,
};

struct OwnedView<T: R2lTensor> {
    states: Vec<T>,
    next_states: Vec<T>,
    actions: Vec<T>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl<T: R2lTensor> OwnedView<T> {
    fn new(
        states: Vec<T>,
        next_states: Vec<T>,
        actions: Vec<T>,
        rewards: Vec<f32>,
        terminated: Vec<bool>,
        truncated: Vec<bool>,
    ) -> Self {
        Self {
            states,
            next_states,
            actions,
            rewards,
            terminated,
            truncated,
        }
    }
}

enum TrajectoryViewsWrapper<'a, T: R2lTensor> {
    Borrowed(TrajectoryView<'a, T>),
    Owned(OwnedView<T>),
}

impl<'a, T: R2lTensor> TrajectoryViewsWrapper<'a, T> {
    fn from_view<'b, S: R2lTensor + Into<T>, V: TrajectoryBatchT<S>>(
        view: &'b V,
    ) -> TrajectoryViewsWrapper<'b, T> {
        if TypeId::of::<S>() == TypeId::of::<T>() {
            let states = unsafe { std::mem::transmute(view.states()) };
            let next_states = unsafe { std::mem::transmute(view.next_states()) };
            let actions = unsafe { std::mem::transmute(view.actions()) };
            return TrajectoryViewsWrapper::Borrowed(TrajectoryView {
                states,
                next_states,
                actions,
                rewards: view.rewards(),
                terminated: view.terminated(),
                truncated: view.truncated(),
            });
        }
        let states = view.states().iter().map(|v| T::convert(v)).collect();
        let next_states = view.next_states().iter().map(|v| T::convert(v)).collect();
        let actions = view.actions().iter().map(|v| T::convert(v)).collect();
        let rewards = view.rewards().to_vec();
        let terminated = view.terminated().to_vec();
        let truncated = view.truncated().to_vec();
        TrajectoryViewsWrapper::Owned(OwnedView::new(
            states,
            next_states,
            actions,
            rewards,
            terminated,
            truncated,
        ))
    }
}

impl<'a, T: R2lTensor> TrajectoryBatchT<T> for TrajectoryViewsWrapper<'a, T> {
    fn len(&self) -> usize {
        todo!()
    }

    fn is_empty(&self) -> bool {
        todo!()
    }

    fn states(&self) -> &[T] {
        todo!()
    }

    fn next_states(&self) -> &[T] {
        todo!()
    }

    fn actions(&self) -> &[T] {
        todo!()
    }

    fn rewards(&self) -> &[f32] {
        todo!()
    }

    fn terminated(&self) -> &[bool] {
        todo!()
    }

    fn truncated(&self) -> &[bool] {
        todo!()
    }
}
