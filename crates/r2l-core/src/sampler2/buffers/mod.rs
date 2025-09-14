use ringbuffer::RingBuffer;
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::{
    env::{Env, Memory},
    sampler::trajectory_buffers::fixed_size_buffer::FixedSizeTrajectoryBuffer,
    sampler2::Buffer,
};

impl<E: Env> Buffer for FixedSizeTrajectoryBuffer<E> {
    type E = E;

    fn last_state(&self) -> Option<<Self::E as Env>::Tensor> {
        let Some(buffer) = &self.buffer else {
            return None;
        };
        buffer.next_states.back().cloned()
    }

    fn push(&mut self, snapshot: Memory<<Self::E as Env>::Tensor>) {
        let Some(buffer) = &mut self.buffer else {
            panic!()
        };
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        } = snapshot;
        buffer.push(state, next_state, action, reward, terminated, trancuated);
    }

    fn last_state_terminates(&self) -> bool {
        let Some(buffer) = &self.buffer else { panic!() };
        *buffer.terminated.back().unwrap() || *buffer.trancuated.back().unwrap()
    }
}

pub struct RcBufferWrapper<B: Buffer>(pub Rc<RefCell<B>>);

impl<E: Env, B: Buffer<E = E>> Buffer for RcBufferWrapper<B> {
    type E = E;

    fn last_state(&self) -> Option<<Self::E as Env>::Tensor> {
        let buffer = self.0.borrow();
        buffer.last_state()
    }

    fn push(&mut self, snapshot: Memory<<Self::E as Env>::Tensor>) {
        let mut buffer = self.0.borrow_mut();
        buffer.push(snapshot);
    }

    fn last_state_terminates(&self) -> bool {
        let buffer = self.0.borrow();
        buffer.last_state_terminates()
    }
}

pub struct ArcBufferWrapper<B: Buffer>(pub Arc<Mutex<B>>);

impl<E: Env, B: Buffer<E = E>> Buffer for ArcBufferWrapper<B> {
    type E = E;

    fn last_state(&self) -> Option<<Self::E as Env>::Tensor> {
        let buffer = self.0.lock().unwrap();
        buffer.last_state()
    }

    fn push(&mut self, snapshot: Memory<<Self::E as Env>::Tensor>) {
        let mut buffer = self.0.lock().unwrap();
        buffer.push(snapshot);
    }

    fn last_state_terminates(&self) -> bool {
        let buffer = self.0.lock().unwrap();
        buffer.last_state_terminates()
    }
}
