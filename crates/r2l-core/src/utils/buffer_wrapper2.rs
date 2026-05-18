use std::{any::TypeId, borrow::Cow};

use crate::{buffers::TrajectoryContainer2, tensor::R2lTensor};

struct TrajectoryFieldViewer<'a, T: Clone + 'static>(Cow<'a, [T]>);

impl<'a, T: Clone + 'static> TrajectoryFieldViewer<'a, T> {
    fn new<S: 'static>(data: &'a [S]) -> Self
    where
        T: From<&'a S>,
    {
        if TypeId::of::<S>() == TypeId::of::<T>() {
            // SAFETY:
            // TypeId equality for 'static types means S and T are the same concrete type.
            let data = unsafe { &*(data as *const [S] as *const [T]) };
            Self(Cow::Borrowed(data))
        } else {
            Self(Cow::Owned(data.into_iter().map(|x| x.into()).collect()))
        }
    }

    fn as_slice(&self) -> &[T] {
        &self.0
    }
}

pub struct TrajectoryView<'a, T: R2lTensor> {
    states: TrajectoryFieldViewer<'a, T>,
    next_states: TrajectoryFieldViewer<'a, T>,
    actions: TrajectoryFieldViewer<'a, T>,
    rewards: &'a [f32],
    terminated: &'a [bool],
    trancuated: &'a [bool],
}

impl<'a, T: R2lTensor> TrajectoryView<'a, T> {
    fn from_buffer<T2: R2lTensor, B: TrajectoryContainer2<Tensor = T2>>(buffer: &'a B) -> Self
    where
        T: From<&'a T2>,
    {
        let states: TrajectoryFieldViewer<T> = TrajectoryFieldViewer::new(buffer.states().unwrap());
        let next_states: TrajectoryFieldViewer<T> =
            TrajectoryFieldViewer::new(buffer.next_states().unwrap());
        let actions: TrajectoryFieldViewer<T> =
            TrajectoryFieldViewer::new(buffer.actions().unwrap());
        Self {
            states,
            next_states,
            actions,
            rewards: buffer.rewards().unwrap(),
            terminated: buffer.terminated().unwrap(),
            trancuated: buffer.truncated().unwrap(),
        }
    }

    pub fn states(&self) -> &[T] {
        self.states.as_slice()
    }

    pub fn actions(&self) -> &[T] {
        self.actions.as_slice()
    }

    pub fn next_states(&self) -> &[T] {
        self.next_states.as_slice()
    }

    pub fn rewards(&self) -> &[f32] {
        &self.rewards
    }

    pub fn terminated(&self) -> &[bool] {
        &self.terminated
    }

    pub fn trancuated(&self) -> &[bool] {
        self.trancuated
    }
}
