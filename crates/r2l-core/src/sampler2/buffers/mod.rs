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
        todo!()
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
        *self.terminated.back().unwrap() || *self.trancuated.back().unwrap()
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

    fn states(&self) -> Vec<Self::Tensor> {
        let buffer = self.0.borrow();
        buffer.states()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        let buffer = self.0.borrow();
        buffer.next_states()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        let buffer = self.0.borrow();
        buffer.actions()
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
        let mut buffer = self.0.borrow_mut();
        buffer.push(snapshot);
    }

    fn last_state_terminates(&self) -> bool {
        let buffer = self.0.borrow();
        buffer.last_state_terminates()
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

    fn states(&self) -> Vec<Self::Tensor> {
        let buffer = self.0.lock().unwrap();
        buffer.states()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        let buffer = self.0.lock().unwrap();
        buffer.next_states()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        let buffer = self.0.lock().unwrap();
        buffer.actions()
    }

    fn rewards(&self) -> Vec<f32> {
        let buffer = self.0.lock().unwrap();
        buffer.rewards()
    }

    fn terminated(&self) -> Vec<bool> {
        todo!()
    }

    fn trancuated(&self) -> Vec<bool> {
        todo!()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        let mut buffer = self.0.lock().unwrap();
        buffer.push(snapshot);
    }

    fn last_state_terminates(&self) -> bool {
        let buffer = self.0.lock().unwrap();
        buffer.last_state_terminates()
    }
}
