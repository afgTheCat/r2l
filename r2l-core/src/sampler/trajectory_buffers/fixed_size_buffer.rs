use crate::{
    distributions::Distribution,
    env::{Env, SnapShot},
    numeric::Buffer,
    rng::RNG,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Device, Tensor};
use crossbeam::channel::{Receiver, RecvError, SendError, Sender};
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
}

impl<E: Env<Tensor = Buffer>> FixedSizeStateBuffer<E> {
    // TODO: this should be removed once we are done
    pub fn to_rollout_buffer(&mut self, size: usize) -> RolloutBuffer {
        let mut rb = RolloutBuffer::default();
        for idx in 0..size {
            let next_state = self.next_states.dequeue().unwrap();
            let state = self.states.dequeue().unwrap();
            let action = self.action.dequeue().unwrap();
            let reward = self.rewards.dequeue().unwrap();
            let terminated = self.terminated.dequeue().unwrap();
            let truncuated = self.trancuated.dequeue().unwrap();
            rb.states.push(state.to_candle_tensor(&Device::Cpu));
            rb.actions.push(action.to_candle_tensor(&Device::Cpu));
            rb.rewards.push(reward);
            rb.dones.push(terminated || truncuated);
            if idx == size - 1 {
                rb.states.push(next_state.to_candle_tensor(&Device::Cpu));
                rb.last_state = Some(next_state.to_candle_tensor(&Device::Cpu));
            }
        }
        rb
    }

    pub fn to_rollout_buffers2(&self) -> RolloutBuffer {
        let mut rb = RolloutBuffer::default();
        for idx in 0..self.capacity {
            rb.states
                .push(self.states[idx].to_candle_tensor(&Device::Cpu));
            rb.actions
                .push(self.action[idx].to_candle_tensor(&Device::Cpu));
            rb.rewards.push(self.rewards[idx]);
            rb.dones.push(self.terminated[idx] || self.trancuated[idx]);
        }
        let last_state = self
            .next_states
            .back()
            .map(|b| b.to_candle_tensor(&Device::Cpu))
            .unwrap()
            .clone();
        rb.states.push(last_state.clone());
        rb.last_state = Some(last_state);
        rb
    }
}

pub struct FixedSizeTrajectoryBuffer<E: Env> {
    pub env: E,
    pub buffer: Option<FixedSizeStateBuffer<E>>,
}

impl<E: Env<Tensor = Buffer>> FixedSizeTrajectoryBuffer<E> {
    pub fn new(env: E, capacity: usize) -> Self {
        Self {
            env,
            buffer: Some(FixedSizeStateBuffer::new(capacity)),
        }
    }

    // TODO: I guess it would make sense to inject some data in here, right?
    // What we could do is have the hook inserted here and just send that but that's blocking +
    // slow + nighmeighrish
    pub fn step(&mut self, distr: &impl Distribution<Tensor = Tensor>) {
        let Some(buffer) = &mut self.buffer else {
            todo!()
        };
        let state = if let Some(obs) = buffer.next_states.back() {
            obs.clone()
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

    pub fn step_n(&mut self, distr: &impl Distribution<Tensor = Tensor>, steps: usize) {
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

    pub fn set_buffer(&mut self, buffer: FixedSizeStateBuffer<E>) {
        self.buffer = Some(buffer)
    }

    pub fn send_buffer(
        &mut self,
        tx: Sender<FixedSizeStateBuffer<E>>,
    ) -> Result<(), SendError<FixedSizeStateBuffer<E>>> {
        let Some(buffer) = self.buffer.take() else {
            todo!()
        };
        tx.send(buffer)
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
