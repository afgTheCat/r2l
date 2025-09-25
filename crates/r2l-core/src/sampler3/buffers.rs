use crate::{
    env::{Env, Memory},
    sampler2::{Buffer, CollectionBound},
};
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
    sync::{Arc, Mutex, MutexGuard},
};

pub struct FixedSizeStateBuffer<E: Env> {
    capacity: usize,
    pub states: AllocRingBuffer<E::Tensor>,
    pub next_states: AllocRingBuffer<E::Tensor>,
    pub rewards: AllocRingBuffer<f32>,
    pub action: AllocRingBuffer<E::Tensor>,
    pub terminated: AllocRingBuffer<bool>,
    pub trancuated: AllocRingBuffer<bool>,
}

impl<E: Env> FixedSizeStateBuffer<E> {
    pub fn push(
        &mut self,
        state: E::Tensor,
        next_state: E::Tensor,
        action: E::Tensor,
        reward: f32,
        terminated: bool,
        trancuated: bool,
    ) {
        self.states.enqueue(state);
        self.next_states.enqueue(next_state);
        self.action.enqueue(action);
        self.rewards.enqueue(reward);
        self.terminated.enqueue(terminated);
        self.trancuated.enqueue(trancuated);
    }
}

impl<E: Env> Buffer for FixedSizeStateBuffer<E> {
    type Tensor = <E as Env>::Tensor;

    fn states(&self) -> Vec<Self::Tensor> {
        self.states.iter().cloned().collect()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        self.next_states.iter().cloned().collect()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        self.action.iter().cloned().collect()
    }

    fn rewards(&self) -> Vec<f32> {
        self.rewards.iter().cloned().collect()
    }

    fn terminated(&self) -> Vec<bool> {
        self.terminated.iter().cloned().collect()
    }

    fn trancuated(&self) -> Vec<bool> {
        self.trancuated.iter().cloned().collect()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        } = snapshot;
        self.push(state, next_state, action, reward, terminated, trancuated);
    }

    fn last_state_terminates(&self) -> bool {
        *self.terminated.back().unwrap() || *self.trancuated.back().unwrap()
    }

    fn build(collection_bound: CollectionBound) -> Self {
        todo!()
    }
}

pub struct VariableSizedStateBuffer<E: Env> {
    pub states: Vec<E::Tensor>,
    pub next_states: Vec<E::Tensor>,
    pub rewards: Vec<f32>,
    pub action: Vec<E::Tensor>,
    pub terminated: Vec<bool>,
    pub trancuated: Vec<bool>,
}

// TODO: for some reason the Default proc macro trips up the compiler. We should investigate this
// in the future.
impl<E: Env> Default for VariableSizedStateBuffer<E> {
    fn default() -> Self {
        Self {
            states: vec![],
            next_states: vec![],
            rewards: vec![],
            action: vec![],
            terminated: vec![],
            trancuated: vec![],
        }
    }
}

impl<E: Env> VariableSizedStateBuffer<E> {
    pub fn push(
        &mut self,
        state: E::Tensor,
        next_state: E::Tensor,
        action: E::Tensor,
        reward: f32,
        terminated: bool,
        trancuated: bool,
    ) {
        self.states.push(state);
        self.next_states.push(next_state);
        self.action.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.trancuated.push(trancuated);
    }
}

impl<E: Env> Buffer for VariableSizedStateBuffer<E> {
    type Tensor = <E as Env>::Tensor;

    fn states(&self) -> Vec<Self::Tensor> {
        self.states.iter().cloned().collect()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        self.next_states.iter().cloned().collect()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        self.action.iter().cloned().collect()
    }

    fn rewards(&self) -> Vec<f32> {
        self.rewards.clone()
    }

    fn terminated(&self) -> Vec<bool> {
        todo!()
    }

    fn trancuated(&self) -> Vec<bool> {
        todo!()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        } = snapshot;
        self.push(state, next_state, action, reward, terminated, trancuated);
    }

    fn last_state_terminates(&self) -> bool {
        *self.terminated.last().unwrap() || *self.trancuated.last().unwrap()
    }

    fn build(collection_bound: CollectionBound) -> Self {
        todo!()
    }
}

pub struct RcBufferWrapper<B: Buffer>(pub Rc<RefCell<B>>);

impl<'a, B: Buffer> RcBufferWrapper<B> {
    pub fn new(buffer: B) -> Self {
        Self(Rc::new(RefCell::new(buffer)))
    }

    pub fn build(collection_bound: CollectionBound) -> Self {
        Self(Rc::new(RefCell::new(B::build(collection_bound))))
    }

    pub fn buffer(&'a self) -> Ref<'a, B> {
        self.0.borrow()
    }

    pub fn buffer_mut(&'a self) -> RefMut<'a, B> {
        self.0.borrow_mut()
    }
}

impl<B: Buffer> Clone for RcBufferWrapper<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(Debug)]
pub struct ArcBufferWrapper<B: Buffer>(pub Arc<Mutex<B>>);

impl<'a, B: Buffer> ArcBufferWrapper<B> {
    pub fn new(buffer: Arc<Mutex<B>>) -> Self {
        Self(buffer)
    }

    pub fn buffer(&'a self) -> MutexGuard<'a, B> {
        self.0.lock().unwrap()
    }
}

impl<B: Buffer> Clone for ArcBufferWrapper<B> {
    fn clone(&self) -> Self {
        ArcBufferWrapper(self.0.clone())
    }
}

// TODO: whether this is the right construct or not remains to be seen
pub enum BufferStack<B: Buffer> {
    RefCounted(Vec<RcBufferWrapper<B>>),
    AtomicRefCounted(Vec<ArcBufferWrapper<B>>),
}
