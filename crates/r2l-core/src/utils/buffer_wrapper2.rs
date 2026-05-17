use std::marker::PhantomData;

use crate::{
    buffers::{Memory, TrajectoryContainer2, fixed_size2::FixedSizeStateBuffer2},
    tensor::R2lTensor,
};

pub struct BorrowedWrapper<
    'a,
    S: R2lTensor,
    T: R2lTensor + From<T>,
    B: TrajectoryContainer2<Tensor = S>,
> {
    data: &'a B,
    _phantom: PhantomData<(S, T)>,
}

fn cast_slice<Src, Dst>(x: &[Src]) -> &[Dst] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

impl<'a, S: R2lTensor, T: R2lTensor + From<T>, B: TrajectoryContainer2<Tensor = S>>
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

    fn states(&self) -> Option<&[T]> {
        self.data.states().map(cast_slice)
    }

    fn next_states(&self) -> Option<&[T]> {
        self.data.next_states().map(cast_slice)
    }

    fn actions(&self) -> Option<&[T]> {
        self.data.actions().map(cast_slice)
    }

    fn rewards(&self) -> Option<&[f32]> {
        self.data.rewards()
    }

    fn terminated(&self) -> Option<&[bool]> {
        self.data.terminated()
    }

    fn truncated(&self) -> &[bool] {
        self.data.truncated()
    }
}

pub enum BufferWrapper<
    'a,
    S: R2lTensor,
    T: R2lTensor + From<T>,
    B: TrajectoryContainer2<Tensor = S>,
> {
    Borrowed(BorrowedWrapper<'a, S, T, B>),
    Cloned(Box<FixedSizeStateBuffer2<T>>),
}

impl<'a, S: R2lTensor, T: R2lTensor + From<S>, B: TrajectoryContainer2<Tensor = S>>
    BufferWrapper<'a, S, T, B>
{
    pub fn new(buffer: &'a B) -> Self {
        let len = buffer.len();
        if std::any::TypeId::of::<S>() == std::any::TypeId::of::<T>() {
            let borrowed = BorrowedWrapper::try_new(buffer).unwrap();
            Self::Borrowed(borrowed)
        } else {
            let states = buffer
                .states()
                .expect("BufferWrapper2::new requires completed rollout states");
            let next_states = buffer
                .next_states()
                .expect("BufferWrapper2::new requires completed rollout next_states");
            let actions = buffer
                .actions()
                .expect("BufferWrapper2::new requires completed rollout actions");
            let rewards = buffer
                .rewards()
                .expect("BufferWrapper2::new requires completed rollout rewards");
            let terminated = buffer
                .terminated()
                .expect("BufferWrapper2::new requires completed rollout terminated flags");
            let truncated = buffer.truncated();

            let mut out = FixedSizeStateBuffer2::new(len);
            for i in 0..len {
                out.push(Memory {
                    state: T::from(states[i].clone()),
                    next_state: T::from(next_states[i].clone()),
                    action: T::from(actions[i].clone()),
                    reward: rewards[i],
                    terminated: terminated[i],
                    truncated: truncated[i],
                });
            }
            BufferWrapper::Cloned(Box::new(out))
        }
    }
}

impl<'a, S: R2lTensor, T: R2lTensor + From<T>, B: TrajectoryContainer2<Tensor = S>>
    TrajectoryContainer2 for BufferWrapper<'a, S, T, B>
{
    type Tensor = T;

    fn len(&self) -> usize {
        match self {
            Self::Borrowed(b) => b.data.len(),
            Self::Cloned(b) => b.len(),
        }
    }

    fn states(&self) -> Option<&[Self::Tensor]> {
        match self {
            Self::Borrowed(b) => b.states(),
            Self::Cloned(b) => b.states(),
        }
    }

    fn next_states(&self) -> Option<&[Self::Tensor]> {
        match self {
            Self::Borrowed(b) => b.next_states(),
            Self::Cloned(b) => b.next_states(),
        }
    }

    fn actions(&self) -> Option<&[Self::Tensor]> {
        match self {
            Self::Borrowed(b) => b.actions(),
            Self::Cloned(b) => b.actions(),
        }
    }

    fn rewards(&self) -> Option<&[f32]> {
        match self {
            Self::Borrowed(b) => b.rewards(),
            Self::Cloned(b) => b.rewards(),
        }
    }

    fn terminated(&self) -> Option<&[bool]> {
        match self {
            Self::Borrowed(b) => b.terminated(),
            Self::Cloned(b) => b.terminated(),
        }
    }

    fn truncated(&self) -> &[bool] {
        match self {
            Self::Borrowed(b) => b.truncated(),
            Self::Cloned(b) => b.truncated(),
        }
    }

    fn begin_rollout(&mut self) {
        match self {
            Self::Borrowed(_) => {
                panic!("BufferWrapper2 does not support mutating borrowed buffers")
            }
            Self::Cloned(b) => b.begin_rollout(),
        }
    }

    fn push(&mut self, memory: Memory<Self::Tensor>) {
        match self {
            Self::Borrowed(_) => {
                panic!("BufferWrapper2 does not support mutating borrowed buffers")
            }
            Self::Cloned(b) => b.push(memory),
        }
    }

    fn pop(&mut self) -> Option<Memory<Self::Tensor>> {
        match self {
            Self::Borrowed(_) => {
                panic!("BufferWrapper2 does not support mutating borrowed buffers")
            }
            Self::Cloned(b) => b.pop(),
        }
    }
}
