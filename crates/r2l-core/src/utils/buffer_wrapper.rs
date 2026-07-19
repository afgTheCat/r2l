use std::any::TypeId;

use crate::{
    buffers::{TrajectoryBatch, buffer::TrajectoryView},
    tensor::R2lTensor,
};

pub struct OwnedView<T: R2lTensor, S: Clone + Send + Sync + 'static> {
    states: Vec<T>,
    next_states: Vec<T>,
    actions: Vec<T>,
    actor_states: Vec<Option<S>>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl<T: R2lTensor, S: Clone + Send + Sync + 'static> OwnedView<T, S> {
    fn new(
        states: Vec<T>,
        next_states: Vec<T>,
        actions: Vec<T>,
        actor_states: Vec<Option<S>>,
        rewards: Vec<f32>,
        terminated: Vec<bool>,
        truncated: Vec<bool>,
    ) -> Self {
        Self {
            states,
            next_states,
            actions,
            actor_states,
            rewards,
            terminated,
            truncated,
        }
    }
}

pub enum TrajectoryViewsWrapper<'a, T: R2lTensor, S: Clone + Send + Sync + 'static = ()> {
    Borrowed(TrajectoryView<'a, T, S>),
    Owned(OwnedView<T, S>),
}

impl<'a, T: R2lTensor, S: Clone + Send + Sync + 'static> TrajectoryViewsWrapper<'a, T, S> {
    pub fn from_view<'b, U: R2lTensor>(
        view: &'b TrajectoryView<'b, U, S>,
    ) -> TrajectoryViewsWrapper<'b, T, S> {
        if TypeId::of::<U>() == TypeId::of::<T>() {
            let states = unsafe { std::mem::transmute::<&[U], &[T]>(view.states()) };
            let next_states = unsafe { std::mem::transmute::<&[U], &[T]>(view.next_states()) };
            let actions = unsafe { std::mem::transmute::<&[U], &[T]>(view.actions()) };
            return TrajectoryViewsWrapper::Borrowed(TrajectoryView {
                states,
                next_states,
                actions,
                actor_states: view.actor_states(),
                rewards: view.rewards(),
                terminated: view.terminated(),
                truncated: view.truncated(),
            });
        }
        let states = view.states().iter().map(|v| T::convert(v)).collect();
        let next_states = view.next_states().iter().map(|v| T::convert(v)).collect();
        let actions = view.actions().iter().map(|v| T::convert(v)).collect();
        let actor_states = view.actor_states().to_vec();
        let rewards = view.rewards().to_vec();
        let terminated = view.terminated().to_vec();
        let truncated = view.truncated().to_vec();
        TrajectoryViewsWrapper::Owned(OwnedView::new(
            states,
            next_states,
            actions,
            actor_states,
            rewards,
            terminated,
            truncated,
        ))
    }
}

impl<'a, T: R2lTensor, S: Clone + Send + Sync + 'static> TrajectoryBatch<T>
    for TrajectoryViewsWrapper<'a, T, S>
{
    type State = S;

    fn len(&self) -> usize {
        match self {
            Self::Borrowed(t) => t.len(),
            Self::Owned(o) => o.states.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Borrowed(t) => t.is_empty(),
            Self::Owned(o) => o.states.is_empty(),
        }
    }

    fn states(&self) -> &[T] {
        match self {
            Self::Borrowed(t) => t.states(),
            Self::Owned(o) => &o.states,
        }
    }

    fn next_states(&self) -> &[T] {
        match self {
            Self::Borrowed(t) => t.next_states(),
            Self::Owned(o) => &o.next_states,
        }
    }

    fn actions(&self) -> &[T] {
        match self {
            Self::Borrowed(t) => t.actions(),
            Self::Owned(o) => &o.actions,
        }
    }

    fn actor_states(&self) -> &[Option<Self::State>] {
        match self {
            Self::Borrowed(t) => t.actor_states(),
            Self::Owned(o) => &o.actor_states,
        }
    }

    fn rewards(&self) -> &[f32] {
        match self {
            Self::Borrowed(t) => t.rewards(),
            Self::Owned(o) => &o.rewards,
        }
    }

    fn terminated(&self) -> &[bool] {
        match self {
            Self::Borrowed(t) => t.terminated(),
            Self::Owned(o) => &o.terminated,
        }
    }

    fn truncated(&self) -> &[bool] {
        match self {
            Self::Borrowed(t) => t.truncated(),
            Self::Owned(o) => &o.truncated,
        }
    }
}
