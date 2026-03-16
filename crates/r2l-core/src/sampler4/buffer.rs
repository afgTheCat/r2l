use std::{any::TypeId, marker::PhantomData, sync::Arc};

use crate::{
    distributions::Policy,
    env::{Env, RolloutMode, SnapShot},
    rng::RNG,
    tensor::{R2lBuffer, R2lTensor},
};
use rand::Rng;
use ringbuffer::AllocRingBuffer;

// What is bad here is that we have a memory
pub struct Memory<T> {
    pub state: T,
    pub next_state: T,
    pub action: T,
    pub reward: f32,
    pub terminated: bool,
    pub trancuated: bool,
}

pub trait TrajectoryContainer: Send + Sync + 'static {
    type Tensor: R2lTensor;

    fn states(&self) -> Vec<Self::Tensor>;

    fn next_states(&self) -> Vec<Self::Tensor>;

    fn actions(&self) -> Vec<Self::Tensor>;

    fn rewards(&self) -> Vec<f32>;

    fn terminated(&self) -> Vec<bool>;

    fn trancuated(&self) -> Vec<bool>;

    fn push(&mut self, snapshot: Memory<Self::Tensor>);
}

struct StepBound<T: R2lTensor> {
    bound: usize,
    _phantom: PhantomData<T>,
}

impl<T: R2lTensor> TrajectoryBound for StepBound<T> {
    type Tensor = T;
    type Container = FixedSizeStateBuffer<T>;

    fn to_container(&self) -> Self::Container {
        FixedSizeStateBuffer::new(self.bound)
    }

    fn method(&self) -> RolloutMode {
        RolloutMode::StepBound {
            n_steps: self.bound,
        }
    }
}

struct EpisodeBound(usize);

pub trait TrajectoryBound: Send + Sync {
    type Tensor: R2lTensor;
    // The caontainer type that is able to work with the given trajectory bound
    type Container: TrajectoryContainer<Tensor = Self::Tensor>;

    fn to_container(&self) -> Self::Container;
    fn method(&self) -> RolloutMode;
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

impl<T: R2lTensor> FixedSizeStateBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            states: AllocRingBuffer::new(capacity),
            next_states: AllocRingBuffer::new(capacity),
            rewards: AllocRingBuffer::new(capacity),
            action: AllocRingBuffer::new(capacity),
            terminated: AllocRingBuffer::new(capacity),
            trancuated: AllocRingBuffer::new(capacity),
        }
    }
}

impl<T: R2lTensor> TrajectoryContainer for FixedSizeStateBuffer<T> {
    type Tensor = T;

    fn states(&self) -> Vec<Self::Tensor> {
        todo!()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        todo!()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        todo!()
    }

    fn rewards(&self) -> Vec<f32> {
        todo!()
    }

    fn terminated(&self) -> Vec<bool> {
        todo!()
    }

    fn trancuated(&self) -> Vec<bool> {
        todo!()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        todo!()
    }
}

// Note: should we set a default here? Probably
// TODO:: not a 100% sure if we need this
pub struct BufferS<D: TrajectoryContainer = FixedSizeStateBuffer<R2lBuffer>> {
    data: D,
}

impl<T: R2lTensor, D: TrajectoryContainer<Tensor = T>> BufferS<D> {
    pub fn new(data: D) -> Self {
        Self { data }
    }

    pub fn last_state(&self) -> Option<T> {
        todo!()
    }

    pub fn last_state_terminates(&self) -> bool {
        todo!()
    }

    pub fn step<E: Env<Tensor = T>>(
        &mut self,
        env: &mut E,
        distr: &Box<dyn Policy<Tensor = T>>,
        last_state: Option<T>,
    ) {
        let state = if let Some(state) = self.last_state() {
            state
        } else if let Some(last_state) = last_state {
            last_state
        } else {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            env.reset(seed).unwrap()
        };
        let action = distr.get_action(state.clone()).unwrap();
        let SnapShot {
            state: mut next_state,
            reward,
            terminated,
            trancuated,
        } = env.step(action.clone()).unwrap();
        let done = terminated || trancuated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            next_state = env.reset(seed).unwrap();
        }
        self.data.push(Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        });
    }
}

// It is questionable how would this work.
pub struct BufferS2<T: R2lTensor> {
    data: Arc<dyn TrajectoryContainer<Tensor = T>>,
}

impl<T: R2lTensor> Clone for BufferS2<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<T: R2lTensor> BufferS2<T> {
    pub fn new<D: TrajectoryContainer<Tensor = T>>(data: D) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    pub fn from_arc(data: Arc<dyn TrajectoryContainer<Tensor = T>>) -> Self {
        Self { data }
    }
}

struct MappedTrajectoryContainer<T: R2lTensor, U: R2lTensor> {
    inner: Arc<dyn TrajectoryContainer<Tensor = T>>,
    _phantom: PhantomData<U>,
}

impl<T: R2lTensor, U: R2lTensor> TrajectoryContainer for MappedTrajectoryContainer<T, U>
where
    U: From<T>,
{
    type Tensor = U;

    fn states(&self) -> Vec<Self::Tensor> {
        self.inner.states().into_iter().map(U::from).collect()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        self.inner.next_states().into_iter().map(U::from).collect()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        self.inner.actions().into_iter().map(U::from).collect()
    }

    fn rewards(&self) -> Vec<f32> {
        self.inner.rewards()
    }

    fn terminated(&self) -> Vec<bool> {
        self.inner.terminated()
    }

    fn trancuated(&self) -> Vec<bool> {
        self.inner.trancuated()
    }

    fn push(&mut self, _snapshot: Memory<Self::Tensor>) {
        panic!("push is not supported on mapped trajectory containers");
    }
}

pub enum BufferWrapper<'a, T: R2lTensor> {
    Borrowed(&'a [BufferS2<T>]),
    Owned(Vec<BufferS2<T>>),
}

impl<'a, T: R2lTensor> AsRef<[BufferS2<T>]> for BufferWrapper<'a, T> {
    fn as_ref(&self) -> &[BufferS2<T>] {
        match self {
            Self::Borrowed(buffers) => buffers,
            Self::Owned(buffers) => buffers.as_slice(),
        }
    }
}

// Ok this is kinda stupid, since we still need to convert between two
pub fn buffer_wrapper<'a, T: R2lTensor, U: R2lTensor + From<T>>(
    buffers: &'a impl AsRef<[BufferS2<T>]>,
) -> BufferWrapper<'a, U> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        todo!()
    } else {
        // buffers.as_ref().iter().map(|b| {});
        todo!()
    }

    // let mapped = buffers
    //     .as_ref()
    //     .iter()
    //     .map(|buffer| {
    //         let mapped = MappedTrajectoryContainer::<T, U> {
    //             inner: buffer.data.clone(),
    //             _phantom: PhantomData,
    //         };
    //         BufferS2::from_arc(Arc::new(mapped))
    //     })
    //     .collect::<Vec<_>>();

    // BufferWrapper::Owned(mapped)
}
