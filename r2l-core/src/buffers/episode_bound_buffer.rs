use crate::{
    distributions::Distribution,
    env::{Env, SnapShot, SnapShot2},
    numeric::Buffer,
    rng::RNG,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Device, Tensor};
use crossbeam::channel::{Receiver, RecvError, SendError, Sender};
use interprocess::TryClone;
use rand::Rng;
use ringbuffer::{AllocRingBuffer, RingBuffer};

pub struct StateBuffer<E: Env> {
    states: AllocRingBuffer<E::Tensor>,
    next_states: AllocRingBuffer<E::Tensor>,
    rewards: AllocRingBuffer<f32>,
    action: AllocRingBuffer<E::Tensor>,
    terminated: AllocRingBuffer<bool>,
    trancuated: AllocRingBuffer<bool>,
}

impl<E: Env> StateBuffer<E> {
    pub fn push_snapshot(
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

// impl<E: Env<Tensor = Tensor>> StateBuffer<E> {
//     // TODO: this should be removed once we are done
//     pub fn to_rollout_buffer(&mut self, size: usize) -> RolloutBuffer {
//         let mut rb = RolloutBuffer::default();
//         for idx in 0..size {
//             let next_state = self.next_states.dequeue().unwrap();
//             let state = self.states.dequeue().unwrap();
//             let action = self.action.dequeue().unwrap();
//             let reward = self.rewards.dequeue().unwrap();
//             let terminated = self.terminated.dequeue().unwrap();
//             let truncuated = self.trancuated.dequeue().unwrap();
//             if idx == 0 {
//                 rb.last_state = Some(next_state.clone());
//                 rb.states.push(next_state);
//             }
//             rb.states.push(state);
//             rb.actions.push(action);
//             rb.rewards.push(reward);
//             rb.dones.push(terminated || truncuated);
//         }
//         rb.states.reverse();
//         rb.actions.reverse();
//         rb.rewards.reverse();
//         rb.dones.reverse();
//         rb
//     }
// }

impl<E: Env<Tensor = Buffer>> StateBuffer<E> {
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
            if idx == 0 {
                rb.last_state = Some(next_state.to_candle_tensor(&Device::Cpu));
                rb.states.push(next_state.to_candle_tensor(&Device::Cpu));
            }
            rb.states.push(state.to_candle_tensor(&Device::Cpu));
            rb.actions.push(action.to_candle_tensor(&Device::Cpu));
            rb.rewards.push(reward);
            rb.dones.push(terminated || truncuated);
        }
        rb.states.reverse();
        rb.actions.reverse();
        rb.rewards.reverse();
        rb.dones.reverse();
        rb
    }
}

pub struct StepBoundBuffer<E: Env> {
    pub env: E,
    pub buffer: Option<StateBuffer<E>>,
}

impl<E: Env<Tensor = Buffer>> StepBoundBuffer<E> {
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
            // TODO: get the seed from core
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            self.env.reset(seed)
        };
        let action = distr
            .get_action(state.to_candle_tensor(&Device::Cpu).clone())
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
        buffer.push_snapshot(
            state,
            next_state,
            Buffer::from_candle_tensor(&action),
            reward,
            terminated,
            trancuated,
        );
    }

    pub fn move_buffer(&mut self) -> StateBuffer<E> {
        if let Some(buffer) = self.buffer.take() {
            buffer
        } else {
            todo!()
        }
    }

    pub fn set_buffer(&mut self, buffer: StateBuffer<E>) {
        self.buffer = Some(buffer)
    }

    pub fn send_buffer(
        &mut self,
        tx: Sender<StateBuffer<E>>,
    ) -> Result<(), SendError<StateBuffer<E>>> {
        let Some(buffer) = self.buffer.take() else {
            todo!()
        };
        tx.send(buffer)
    }

    pub fn receive_buffer(&mut self, rx: Receiver<StateBuffer<E>>) -> Result<(), RecvError> {
        let buffer = rx.recv()?;
        self.buffer = Some(buffer);
        Ok(())
    }
}
