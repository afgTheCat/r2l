pub mod fixed_size;
pub mod wrapper;

use itertools::izip;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::marker::PhantomData;

use crate::{
    env::{Env, SnapShot},
    sampler5::RolloutMode,
    tensor::R2lTensor,
};

pub struct Memory<T> {
    pub state: T,
    pub next_state: T,
    pub action: T,
    pub reward: f32,
    pub terminated: bool,
    pub trancuated: bool,
}

impl<T> Memory<T> {
    pub fn terminates(&self) -> bool {
        self.terminated || self.trancuated
    }
}

pub trait TrajectoryContainer: Sync {
    type Tensor: R2lTensor;

    fn len(&self) -> usize;

    fn states(&self) -> impl Iterator<Item = &Self::Tensor>;

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor>;

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor>;

    fn rewards(&self) -> impl Iterator<Item = f32>;

    fn terminated(&self) -> impl Iterator<Item = bool>;

    fn trancuated(&self) -> impl Iterator<Item = bool>;

    // TODO: warning, this clones the states, probably not good for perf
    // use it only if cloning is neccesarry!
    fn memories(&self) -> impl Iterator<Item = Memory<Self::Tensor>> {
        izip!(
            self.states(),
            self.next_states(),
            self.actions(),
            self.rewards(),
            self.terminated(),
            self.trancuated()
        )
        .map(
            |(state, next_state, action, reward, terminated, trancuated)| Memory {
                state: state.clone(),
                next_state: next_state.clone(),
                action: action.clone(),
                reward,
                terminated,
                trancuated,
            },
        )
    }
}

pub trait ExpandableTrajectoryContainer: TrajectoryContainer + Send + 'static {
    fn push(&mut self, memory: Memory<Self::Tensor>);
}

pub trait TrajectoryBound: Send + Sync {
    type Tensor: R2lTensor;
    // The caontainer type that is able to work with the given trajectory bound
    type Container: ExpandableTrajectoryContainer<Tensor = Self::Tensor>;

    fn to_container(&self) -> Self::Container;
    fn method(&self) -> RolloutMode;
}
