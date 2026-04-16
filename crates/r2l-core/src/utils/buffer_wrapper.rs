use std::marker::PhantomData;

use crate::{
    buffers::{
        ExpandableTrajectoryContainer, Memory, TrajectoryContainer, fix_sized::FixedSizeStateBuffer,
    },
    tensor::R2lTensor,
};

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

pub struct BorrowedWrapper<
    'a,
    S: R2lTensor,
    T: R2lTensor + From<T>,
    B: TrajectoryContainer<Tensor = S>,
> {
    data: &'a B,
    _phantom: PhantomData<(S, T)>,
}

fn cast_ref<Src, Dst>(x: &Src) -> &Dst {
    unsafe { &*(x as *const Src as *const Dst) }
}

impl<'a, S: R2lTensor, T: R2lTensor + From<T>, B: TrajectoryContainer<Tensor = S>>
    BorrowedWrapper<'a, S, T, B>
{
    fn try_new(data: &'a B) -> Option<Self> {
        if std::any::TypeId::of::<S>() == std::any::TypeId::of::<T>() {
            Some(Self {
                data,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }

    fn states(&self) -> TensorIter<'_, T> {
        TensorIter::new(self.data.states().map(cast_ref))
    }

    fn next_states(&self) -> TensorIter<'_, T> {
        TensorIter::new(self.data.next_states().map(cast_ref))
    }

    fn actions(&self) -> TensorIter<'_, T> {
        TensorIter::new(self.data.actions().map(cast_ref))
    }

    fn rewards(&self) -> F32Iter<'_> {
        F32Iter::new(self.data.rewards())
    }

    fn terminated(&self) -> BoolIter<'_> {
        BoolIter::new(self.data.terminated())
    }

    fn trancuated(&self) -> BoolIter<'_> {
        BoolIter::new(self.data.trancuated())
    }
}

pub enum BufferWrapper<'a, S: R2lTensor, T: R2lTensor + From<T>, B: TrajectoryContainer<Tensor = S>>
{
    Borrowed(BorrowedWrapper<'a, S, T, B>),
    Cloned(Box<FixedSizeStateBuffer<T>>),
}

impl<'a, S: R2lTensor, T: R2lTensor + From<S>, B: TrajectoryContainer<Tensor = S>>
    BufferWrapper<'a, S, T, B>
{
    pub fn new(buffer: &'a B) -> Self {
        let len = buffer.len();
        if std::any::TypeId::of::<S>() == std::any::TypeId::of::<T>() {
            let borrowed = BorrowedWrapper::try_new(buffer).unwrap();
            Self::Borrowed(borrowed)
        } else {
            let mut out = FixedSizeStateBuffer::new(len);
            // let state = T::from(buffer.states().next().unwrap().clone());
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
            BufferWrapper::Cloned(Box::new(out))
        }
    }
}

impl<'a, S: R2lTensor, T: R2lTensor + From<T>, B: TrajectoryContainer<Tensor = S>>
    TrajectoryContainer for BufferWrapper<'a, S, T, B>
{
    type Tensor = T;

    fn len(&self) -> usize {
        match self {
            Self::Borrowed(b) => b.data.len(),
            Self::Cloned(b) => b.len(),
        }
    }

    fn states(&self) -> impl Iterator<Item = &Self::Tensor> {
        match self {
            Self::Borrowed(b) => b.states(),
            Self::Cloned(b) => TensorIter::new(b.states()),
        }
    }

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor> {
        match self {
            Self::Borrowed(b) => b.next_states(),
            Self::Cloned(b) => TensorIter::new(b.next_states()),
        }
    }

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor> {
        match self {
            Self::Borrowed(b) => b.actions(),
            Self::Cloned(b) => TensorIter::new(b.actions()),
        }
    }

    fn rewards(&self) -> impl Iterator<Item = f32> {
        match self {
            Self::Borrowed(b) => b.rewards(),
            Self::Cloned(b) => F32Iter::new(b.rewards()),
        }
    }

    fn terminated(&self) -> impl Iterator<Item = bool> {
        match self {
            Self::Borrowed(b) => b.terminated(),
            Self::Cloned(b) => BoolIter::new(b.terminated()),
        }
    }

    fn trancuated(&self) -> impl Iterator<Item = bool> {
        match self {
            Self::Borrowed(b) => b.trancuated(),
            Self::Cloned(b) => BoolIter::new(b.trancuated()),
        }
    }
}
