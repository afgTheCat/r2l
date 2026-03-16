use std::marker::PhantomData;

use itertools::izip;
use ringbuffer::{AllocRingBuffer, RingBuffer};

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

struct TensorIter<'a, T: R2lTensor> {
    iterator: Box<dyn Iterator<Item = &'a T> + 'a>,
}

impl<'a, T: R2lTensor> TensorIter<'a, T> {
    fn new<I>(iterator: I) -> Self
    where
        I: Iterator<Item = &'a T> + 'a,
    {
        Self {
            iterator: Box::new(iterator),
        }
    }
}

impl<'a, T: R2lTensor> Iterator for TensorIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next()
    }
}

pub trait TrajectoryContainer: Send + Sync {
    type Tensor: R2lTensor;

    fn states(&self) -> impl Iterator<Item = &Self::Tensor> + '_;

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor> + '_;

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor> + '_;

    fn rewards(&self) -> impl Iterator<Item = f32>;

    fn terminated(&self) -> impl Iterator<Item = bool>;

    fn trancuated(&self) -> impl Iterator<Item = bool>;

    // TODO: this need not to be
    fn push(&mut self, memory: Memory<Self::Tensor>);

    // TODO: warning, this clones the states, probably not good for perf
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

pub struct FixedSizeStateBuffer<T: R2lTensor> {
    pub capacity: usize,
    pub states: AllocRingBuffer<T>,
    pub next_states: AllocRingBuffer<T>,
    pub rewards: AllocRingBuffer<f32>,
    pub action: AllocRingBuffer<T>,
    pub terminated: AllocRingBuffer<bool>,
    pub trancuated: AllocRingBuffer<bool>,
}

impl<T: R2lTensor> TrajectoryContainer for FixedSizeStateBuffer<T> {
    type Tensor = T;

    fn states(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.states.iter()
    }

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.next_states.iter()
    }

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.action.iter()
    }

    fn rewards(&self) -> impl Iterator<Item = f32> {
        self.rewards.iter().map(|x| *x)
    }

    fn terminated(&self) -> impl Iterator<Item = bool> {
        self.terminated.iter().map(|x| *x)
    }

    fn trancuated(&self) -> impl Iterator<Item = bool> {
        self.trancuated.iter().map(|x| *x)
    }

    fn push(&mut self, memory: Memory<Self::Tensor>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        } = memory;
        self.states.enqueue(state);
        self.next_states.enqueue(next_state);
        self.action.enqueue(action);
        self.rewards.enqueue(reward);
        self.terminated.enqueue(terminated);
        self.trancuated.enqueue(trancuated);
    }
}

enum BufferWrapper<'a, T: R2lTensor, B: TrajectoryContainer<Tensor = T>> {
    Borrowed(&'a B),
    Cloned(FixedSizeStateBuffer<T>),
}

impl<'a, T: R2lTensor, B: TrajectoryContainer<Tensor = T>> TrajectoryContainer
    for BufferWrapper<'a, T, B>
where
    B: 'static,
{
    type Tensor = T;

    fn states(&self) -> impl Iterator<Item = &Self::Tensor> + '_ {
        match self {
            Self::Borrowed(b) => TensorIter::new((*b).states()),
            Self::Cloned(b) => TensorIter::new(b.states()),
        }
    }

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor> + '_ {
        match self {
            Self::Borrowed(b) => TensorIter::new((*b).next_states()),
            Self::Cloned(b) => TensorIter::new(b.next_states()),
        }
    }

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor> + '_ {
        match self {
            Self::Borrowed(b) => TensorIter::new((*b).actions()),
            Self::Cloned(b) => TensorIter::new(b.actions()),
        }
    }

    fn rewards(&self) -> impl Iterator<Item = f32> {
        match self {
            Self::Borrowed(b) => Box::new((*b).rewards()) as Box<dyn Iterator<Item = f32> + '_>,
            Self::Cloned(b) => Box::new(b.rewards()) as Box<dyn Iterator<Item = f32> + '_>,
        }
    }

    fn terminated(&self) -> impl Iterator<Item = bool> {
        match self {
            Self::Borrowed(b) => Box::new((*b).terminated()) as Box<dyn Iterator<Item = bool> + '_>,
            Self::Cloned(b) => Box::new(b.terminated()) as Box<dyn Iterator<Item = bool> + '_>,
        }
    }

    fn trancuated(&self) -> impl Iterator<Item = bool> {
        match self {
            Self::Borrowed(b) => Box::new((*b).trancuated()) as Box<dyn Iterator<Item = bool> + '_>,
            Self::Cloned(b) => Box::new(b.trancuated()) as Box<dyn Iterator<Item = bool> + '_>,
        }
    }

    fn push(&mut self, memory: Memory<Self::Tensor>) {}
}

pub trait TrajectoryBound: Send + Sync {
    type Tensor: R2lTensor;
    // The caontainer type that is able to work with the given trajectory bound
    type Container: TrajectoryContainer<Tensor = Self::Tensor>;

    fn to_container(&self) -> Self::Container;
    fn method(&self) -> RolloutMode;
}
