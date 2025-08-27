use crate::{
    distributions::Distribution,
    env::{Env, SnapShot},
    numeric::Buffer,
    rng::RNG,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Device, Tensor};
use rand::Rng;

pub struct VariableSizedStateBuffer<E: Env> {
    states: Vec<E::Tensor>,
    next_states: Vec<E::Tensor>,
    rewards: Vec<f32>,
    action: Vec<E::Tensor>,
    terminated: Vec<bool>,
    trancuated: Vec<bool>,
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

    pub fn clear(&mut self) {
        self.states.clear();
        self.next_states.clear();
        self.action.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.trancuated.clear();
    }

    pub fn last_state_terminates(&self) -> bool {
        *self.terminated.last().unwrap() || *self.trancuated.last().unwrap()
    }
}

pub struct VariableSizedTrajectoryBuffer<E: Env> {
    pub env: E,
    pub buffer: VariableSizedStateBuffer<E>,
}

impl<E: Env<Tensor = Buffer>> VariableSizedTrajectoryBuffer<E> {
    pub fn new(env: E) -> Self {
        let buffer: VariableSizedStateBuffer<E> = VariableSizedStateBuffer::default();
        Self { env, buffer }
    }

    pub fn step(&mut self, distr: &impl Distribution<Tensor = Tensor>) {
        let buffer = &mut self.buffer;
        let state = if let Some(obs) = buffer.next_states.last() {
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

    pub fn step_with_epiosde_bound(
        &mut self,
        distr: &impl Distribution<Tensor = Tensor>,
        n_steps: usize,
    ) {
        let mut steps_taken = 0;
        loop {
            self.step(distr);
            steps_taken += 1;
            if steps_taken >= n_steps || self.buffer.last_state_terminates() {
                break;
            }
        }
    }

    pub fn to_rollout_buffer(&self) -> RolloutBuffer {
        todo!()
    }
}
