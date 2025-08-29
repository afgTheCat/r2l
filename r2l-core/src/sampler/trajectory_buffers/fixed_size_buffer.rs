use crate::{
    distributions::Distribution,
    env::{Env, SnapShot},
    numeric::Buffer,
    rng::RNG,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Device, Tensor};
use crossbeam::channel::{Receiver, RecvError};
use rand::Rng;
use ringbuffer::{AllocRingBuffer, RingBuffer};

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

    pub fn pop_last_state(&mut self) -> E::Tensor {
        todo!()
    }

    pub fn set_last_state(&mut self, new_state: E::Tensor) {
        self.next_states.enqueue(new_state);
    }

    pub fn pop_last_reward(&mut self) -> f32 {
        todo!()
    }

    pub fn set_last_reward(&mut self, reward: f32) {
        self.rewards.enqueue(reward);
    }

    // TODO: verify if this works, alternatevly there is the old method commented out below
    fn to_rollout_buffers2<D: Clone>(&mut self) -> RolloutBuffer<D>
    where
        E::Tensor: From<D>,
        E::Tensor: Into<D>,
    {
        let mut rb = RolloutBuffer::default();
        for idx in 0..self.capacity {
            let next_state = self.next_states.dequeue().unwrap();
            let state = self.states.dequeue().unwrap();
            let action = self.action.dequeue().unwrap();
            let reward = self.rewards.dequeue().unwrap();
            let terminated = self.terminated.dequeue().unwrap();
            let truncuated = self.trancuated.dequeue().unwrap();
            if idx == self.capacity - 1 {
                rb.states.push(next_state.into());
            }
            rb.states.push(state.into());
            rb.actions.push(action.into());
            rb.rewards.push(reward);
            rb.dones.push(terminated || truncuated);
        }
        rb
    }

    pub fn last_state_terminates(&self) -> bool {
        *self.terminated.back().unwrap() || *self.trancuated.back().unwrap()
    }
}

pub struct FixedSizeTrajectoryBuffer<E: Env> {
    pub env: E,
    pub buffer: Option<FixedSizeStateBuffer<E>>,
    pub last_state: Option<E::Tensor>,
}

impl<E: Env> FixedSizeTrajectoryBuffer<E> {
    pub fn new(env: E, capacity: usize) -> Self {
        Self {
            env,
            buffer: Some(FixedSizeStateBuffer::new(capacity)),
            last_state: None,
        }
    }
}

impl<E: Env<Tensor = Buffer>> FixedSizeTrajectoryBuffer<E> {
    pub fn step<D: Distribution<Tensor = Tensor> + ?Sized>(&mut self, distr: &D) {
        let Some(buffer) = &mut self.buffer else {
            todo!()
        };
        let state = if let Some(state) = buffer.next_states.back() {
            state.clone()
        // last state saved from previous rollout, so that we don't have to reset
        } else if let Some(last_state) = self.last_state.take() {
            last_state
        } else {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            self.env.reset(seed)
        };
        let action = distr
            .get_action(state.to_candle_tensor(&Device::Cpu).unsqueeze(0).unwrap())
            .unwrap();
        let SnapShot {
            state: mut next_state,
            reward,
            terminated,
            trancuated,
        } = self.env.step(Buffer::from_candle_tensor(&action));
        let done = terminated || trancuated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            next_state = self.env.reset(seed);
        }
        buffer.push(
            state,
            next_state,
            Buffer::from_candle_tensor(&action),
            reward,
            terminated,
            trancuated,
        );
    }

    pub fn step_n<D: Distribution<Tensor = Tensor> + ?Sized>(&mut self, distr: &D, steps: usize) {
        for _ in 0..steps {
            self.step(distr);
        }
    }

    pub fn move_buffer(&mut self) -> FixedSizeStateBuffer<E> {
        if let Some(buffer) = self.buffer.take() {
            buffer
        } else {
            todo!()
        }
    }

    pub fn to_rollout_buffer(&mut self) -> RolloutBuffer<Tensor> {
        let Some(buffer) = self.buffer.as_mut() else {
            panic!()
        };
        if !buffer.last_state_terminates() {
            let last_state = buffer.next_states.back().cloned();
            self.last_state = last_state;
        }
        buffer.to_rollout_buffers2()
    }

    pub fn set_buffer(&mut self, buffer: FixedSizeStateBuffer<E>) {
        self.buffer = Some(buffer)
    }

    pub fn receive_buffer(
        &mut self,
        rx: Receiver<FixedSizeStateBuffer<E>>,
    ) -> Result<(), RecvError> {
        let buffer = rx.recv()?;
        self.buffer = Some(buffer);
        Ok(())
    }
}
