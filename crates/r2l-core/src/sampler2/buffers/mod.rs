use crate::{
    env::{Env, Memory},
    sampler::trajectory_buffers::{
        fixed_size_buffer::FixedSizeStateBuffer, variable_size_buffer::VariableSizedStateBuffer,
    },
    sampler2::Buffer,
};
use ringbuffer::RingBuffer;
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

impl<E: Env> Buffer for FixedSizeStateBuffer<E> {
    type Tensor = <E as Env>::Tensor;

    fn last_state(&self) -> Option<Self::Tensor> {
        self.next_states.back().cloned()
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

    fn rewards(&self) -> &[f32] {
        todo!()
    }
}

impl<E: Env> Buffer for VariableSizedStateBuffer<E> {
    type Tensor = <E as Env>::Tensor;

    fn last_state(&self) -> Option<Self::Tensor> {
        self.next_states.last().cloned()
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

    fn rewards(&self) -> &[f32] {
        todo!()
    }
}

pub struct RcBufferWrapper<B: Buffer>(pub Rc<RefCell<B>>);

impl<B: Buffer> RcBufferWrapper<B> {
    pub fn new(ref_cell: Rc<RefCell<B>>) -> Self {
        Self(ref_cell)
    }
}

impl<B: Buffer> Clone for RcBufferWrapper<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Buffer> Buffer for RcBufferWrapper<B> {
    type Tensor = <B as Buffer>::Tensor;

    fn last_state(&self) -> Option<Self::Tensor> {
        let buffer = self.0.borrow();
        buffer.last_state()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        let mut buffer = self.0.borrow_mut();
        buffer.push(snapshot);
    }

    fn last_state_terminates(&self) -> bool {
        let buffer = self.0.borrow();
        buffer.last_state_terminates()
    }

    fn rewards(&self) -> &[f32] {
        todo!()
    }
}

#[derive(Debug)]
pub struct ArcBufferWrapper<B: Buffer>(pub Arc<Mutex<B>>);

impl<B: Buffer> ArcBufferWrapper<B> {
    pub fn new(buffer: Arc<Mutex<B>>) -> Self {
        Self(buffer)
    }
}

impl<B: Buffer> Clone for ArcBufferWrapper<B> {
    fn clone(&self) -> Self {
        ArcBufferWrapper(self.0.clone())
    }
}

impl<B: Buffer> Buffer for ArcBufferWrapper<B> {
    type Tensor = <B as Buffer>::Tensor;

    fn last_state(&self) -> Option<Self::Tensor> {
        let buffer = self.0.lock().unwrap();
        buffer.last_state()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        let mut buffer = self.0.lock().unwrap();
        buffer.push(snapshot);
    }

    fn last_state_terminates(&self) -> bool {
        let buffer = self.0.lock().unwrap();
        buffer.last_state_terminates()
    }

    fn rewards(&self) -> &[f32] {
        todo!()
    }
}
