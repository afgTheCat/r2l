pub mod fixed_size;
pub mod variable_size;
pub mod wrapper;

use crate::{
    sampler::{RolloutMode, buffer::fixed_size::FixedSizeStateBuffer},
    tensor::R2lTensor,
};
use itertools::izip;
use std::marker::PhantomData;

#[derive(Debug)]
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

    fn dones(&self) -> impl Iterator<Item = bool> {
        self.terminated()
            .zip(self.trancuated())
            .map(|(terminated, trancuated)| terminated || trancuated)
    }

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

// TODO: I think maybe we need a pop_last. Or maybe a mut ref to the last
pub trait EditableTrajectoryContainer: TrajectoryContainer {
    fn pop_last_state(&mut self) -> Self::Tensor;

    fn pop_last_reward(&mut self) -> f32;

    fn set_last_state(&mut self, t: Self::Tensor);

    fn set_last_reward(&mut self, r: f32);
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

pub struct StepTrajectoryBound<T: R2lTensor> {
    steps: usize,
    _phantom: PhantomData<T>,
}

impl<T: R2lTensor> StepTrajectoryBound<T> {
    pub fn new(steps: usize) -> Self {
        Self {
            steps,
            _phantom: PhantomData,
        }
    }
}

impl<T: R2lTensor> TrajectoryBound for StepTrajectoryBound<T> {
    type Tensor = T;
    type Container = FixedSizeStateBuffer<T>;

    fn to_container(&self) -> Self::Container {
        FixedSizeStateBuffer::new(self.steps)
    }

    fn method(&self) -> RolloutMode {
        RolloutMode::StepBound {
            n_steps: self.steps,
        }
    }
}
