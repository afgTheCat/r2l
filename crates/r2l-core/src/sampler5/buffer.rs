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

struct F32Iter<'a> {
    iter: Box<dyn Iterator<Item = f32> + 'a>,
}

impl<'a> F32Iter<'a> {
    fn new<I>(iterator: I) -> Self
    where
        I: Iterator<Item = f32> + 'a,
    {
        Self {
            iter: Box::new(iterator),
        }
    }
}

impl<'a> Iterator for F32Iter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

struct BoolIter<'a> {
    iter: Box<dyn Iterator<Item = bool> + 'a>,
}

impl<'a> BoolIter<'a> {
    fn new<I>(iterator: I) -> Self
    where
        I: Iterator<Item = bool> + 'a,
    {
        Self {
            iter: Box::new(iterator),
        }
    }
}

impl<'a> Iterator for BoolIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
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

    fn len(&self) -> usize;

    fn states(&self) -> impl Iterator<Item = &Self::Tensor>;

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor>;

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor>;

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
    pub len: usize,
    pub states: AllocRingBuffer<T>,
    pub next_states: AllocRingBuffer<T>,
    pub rewards: AllocRingBuffer<f32>,
    pub action: AllocRingBuffer<T>,
    pub terminated: AllocRingBuffer<bool>,
    pub trancuated: AllocRingBuffer<bool>,
}

impl<T: R2lTensor> FixedSizeStateBuffer<T> {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            states: AllocRingBuffer::new(len),
            next_states: AllocRingBuffer::new(len),
            rewards: AllocRingBuffer::new(len),
            action: AllocRingBuffer::new(len),
            terminated: AllocRingBuffer::new(len),
            trancuated: AllocRingBuffer::new(len),
        }
    }
}

impl<T: R2lTensor> TrajectoryContainer for FixedSizeStateBuffer<T> {
    type Tensor = T;

    fn len(&self) -> usize {
        self.len
    }

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

struct Borrowed<'a, T: R2lTensor> {
    states: TensorIter<'a, T>,
    next_states: TensorIter<'a, T>,
    actions: TensorIter<'a, T>,
    rewards: F32Iter<'a>,
    terminated: BoolIter<'a>,
    trancuated: BoolIter<'a>,
}

enum BufferWrapper<'a, T: R2lTensor, B: TrajectoryContainer<Tensor = T>> {
    Borrowed(&'a B),
    Cloned(FixedSizeStateBuffer<T>),
}

enum BufferWrapper2<'a, T: R2lTensor> {
    Borrowed(Borrowed<'a, T>),
    Cloned(FixedSizeStateBuffer<T>),
}

fn cast_ref<Src: 'static, Dst: 'static>(x: &Src) -> &Dst {
    debug_assert_eq!(std::any::TypeId::of::<Src>(), std::any::TypeId::of::<Dst>());
    unsafe { &*(x as *const Src as *const Dst) }
}

impl<'a, T: R2lTensor> BufferWrapper2<'a, T> {
    fn new<Src, B>(buffer: &'a B) -> Self
    where
        Src: R2lTensor,
        T: From<Src>,
        B: TrajectoryContainer<Tensor = Src>,
    {
        if std::any::TypeId::of::<Src>() == std::any::TypeId::of::<T>() {
            BufferWrapper2::Borrowed(Borrowed {
                states: TensorIter::new(buffer.states().map(cast_ref::<Src, T>)),
                next_states: TensorIter::new(buffer.next_states().map(cast_ref::<Src, T>)),
                actions: TensorIter::new(buffer.actions().map(cast_ref::<Src, T>)),
                rewards: F32Iter::new(buffer.rewards()),
                terminated: BoolIter::new(buffer.terminated()),
                trancuated: BoolIter::new(buffer.trancuated()),
            })
        } else {
            let len = buffer.len();
            let mut out = FixedSizeStateBuffer::new(len);

            for memory in buffer.memories() {
                out.push(Memory {
                    state: T::from(memory.state),
                    next_state: T::from(memory.next_state),
                    action: T::from(memory.action),
                    reward: memory.reward,
                    terminated: memory.terminated,
                    trancuated: memory.trancuated,
                });
            }

            BufferWrapper2::Cloned(out)
        }
    }
}

impl<'a, T: R2lTensor, B: TrajectoryContainer<Tensor = T>> BufferWrapper<'a, T, B> {
    fn new<E: From<T>>() -> Self {
        todo!()
    }
}

impl<'a, T: R2lTensor, B: TrajectoryContainer<Tensor = T>> TrajectoryContainer
    for BufferWrapper<'a, T, B>
{
    type Tensor = T;

    fn len(&self) -> usize {
        match self {
            Self::Borrowed(b) => b.len(),
            Self::Cloned(b) => b.len(),
        }
    }

    fn states(&self) -> impl Iterator<Item = &Self::Tensor> {
        match self {
            Self::Borrowed(b) => TensorIter::new((*b).states()),
            Self::Cloned(b) => TensorIter::new(b.states()),
        }
    }

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor> {
        match self {
            Self::Borrowed(b) => TensorIter::new((*b).next_states()),
            Self::Cloned(b) => TensorIter::new(b.next_states()),
        }
    }

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor> {
        match self {
            Self::Borrowed(b) => TensorIter::new((*b).actions()),
            Self::Cloned(b) => TensorIter::new(b.actions()),
        }
    }

    fn rewards(&self) -> impl Iterator<Item = f32> {
        match self {
            Self::Borrowed(b) => Box::new((*b).rewards()) as Box<dyn Iterator<Item = f32>>,
            Self::Cloned(b) => Box::new(b.rewards()),
        }
    }

    fn terminated(&self) -> impl Iterator<Item = bool> {
        match self {
            Self::Borrowed(b) => Box::new((*b).terminated()) as Box<dyn Iterator<Item = bool>>,
            Self::Cloned(b) => Box::new(b.terminated()),
        }
    }

    fn trancuated(&self) -> impl Iterator<Item = bool> {
        match self {
            Self::Borrowed(b) => Box::new((*b).trancuated()) as Box<dyn Iterator<Item = bool>>,
            Self::Cloned(b) => Box::new(b.trancuated()),
        }
    }

    // TODO: the buffer we
    fn push(&mut self, memory: Memory<Self::Tensor>) {
        unreachable!()
    }
}

pub trait TrajectoryBound: Send + Sync {
    type Tensor: R2lTensor;
    // The caontainer type that is able to work with the given trajectory bound
    type Container: TrajectoryContainer<Tensor = Self::Tensor>;

    fn to_container(&self) -> Self::Container;
    fn method(&self) -> RolloutMode;
}
