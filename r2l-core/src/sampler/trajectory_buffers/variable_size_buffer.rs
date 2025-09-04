use crate::{
    distributions::Distribution,
    env::{Env, SnapShot},
    rng::RNG,
    utils::rollout_buffer::RolloutBuffer,
};
use rand::Rng;

pub struct VariableSizedStateBuffer<E: Env> {
    states: Vec<E::Tensor>,
    next_states: Vec<E::Tensor>,
    rewards: Vec<f32>,
    action: Vec<E::Tensor>,
    terminated: Vec<bool>,
    trancuated: Vec<bool>,
}

impl<E: Env> VariableSizedStateBuffer<E> {
    fn take_rollout_buffer(&mut self) -> RolloutBuffer<E::Tensor> {
        let mut states = std::mem::take(&mut self.states);
        let actions = std::mem::take(&mut self.action);
        let rewards = std::mem::take(&mut self.rewards);
        let dones = std::mem::take(&mut self.terminated)
            .into_iter()
            .zip(std::mem::take(&mut self.trancuated))
            .map(|(terminated, trancuated)| terminated || trancuated)
            .collect();
        let mut next_states = std::mem::take(&mut self.next_states);
        let last_state = next_states.pop().unwrap();
        states.push(last_state.clone());
        RolloutBuffer {
            states,
            actions,
            rewards,
            dones,
        }
    }
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

    pub fn last_state_terminates(&self) -> bool {
        *self.terminated.last().unwrap() || *self.trancuated.last().unwrap()
    }
}

pub struct VariableSizedTrajectoryBuffer<E: Env> {
    pub env: E,
    // TODO: maybe we could make this optional, signaling that the trajector whether the trajectory
    // buffer currently holds the buffer or not
    pub buffer: VariableSizedStateBuffer<E>,
    pub distr: Option<Box<dyn Distribution<Tensor = E::Tensor>>>,
    pub last_state: Option<E::Tensor>,
}

impl<E: Env> VariableSizedTrajectoryBuffer<E> {
    pub fn new(env: E) -> Self {
        let buffer: VariableSizedStateBuffer<E> = VariableSizedStateBuffer::default();
        Self {
            env,
            buffer,
            distr: None,
            last_state: None,
        }
    }

    pub fn step2(&mut self) {
        let Some(distr) = &mut self.distr else {
            todo!()
        };
        let state = if let Some(state) = self.buffer.next_states.last() {
            state.clone()
        // last state saved from previous rollout, so that we don't have to reset
        } else if let Some(last_state) = self.last_state.take() {
            last_state
        } else {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            self.env.reset(seed).unwrap()
        };
        let action = distr.get_action(state.clone()).unwrap();
        let SnapShot {
            state: mut next_state,
            reward,
            terminated,
            trancuated,
        } = self.env.step(action.clone()).unwrap();
        let done = terminated || trancuated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            next_state = self.env.reset(seed).unwrap();
        }
        self.buffer
            .push(state, next_state, action, reward, terminated, trancuated);
    }

    pub fn step<D: Distribution<Tensor = E::Tensor> + ?Sized>(&mut self, distr: &D) {
        let buffer = &mut self.buffer;
        let state = if let Some(obs) = buffer.next_states.last() {
            obs.clone()
        } else if let Some(last_state) = self.last_state.take() {
            last_state
        } else {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            self.env.reset(seed).unwrap()
        };
        let action = distr.get_action(state.clone()).unwrap();
        let SnapShot {
            state: mut next_state,
            reward,
            terminated,
            trancuated,
        } = self.env.step(action.clone()).unwrap();
        let done = terminated || trancuated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            next_state = self.env.reset(seed).unwrap();
        }
        buffer.push(state, next_state, action, reward, terminated, trancuated);
    }

    pub fn step_with_epiosde_bound<D: Distribution<Tensor = E::Tensor> + ?Sized>(
        &mut self,
        distr: &D,
        n_steps: usize,
    ) {
        let mut steps_taken = 0;
        loop {
            self.step(distr);
            steps_taken += 1;
            if steps_taken >= n_steps && self.buffer.last_state_terminates() {
                break;
            }
        }
    }

    pub fn step_with_epiosde_bound2(&mut self, n_steps: usize) {
        let mut steps_taken = 0;
        loop {
            self.step2();
            steps_taken += 1;
            if steps_taken >= n_steps && self.buffer.last_state_terminates() {
                break;
            }
        }
    }

    pub fn run_episodes<D: Distribution<Tensor = E::Tensor> + ?Sized>(
        &mut self,
        distr: &D,
        episodes: usize,
    ) {
        let mut ep_count = 0;
        while ep_count < episodes {
            self.step(distr);
            if self.buffer.last_state_terminates() {
                ep_count += 1;
            }
        }
    }

    pub fn run_episodes2(&mut self, episodes: usize) {
        let mut ep_count = 0;
        while ep_count < episodes {
            self.step2();
            if self.buffer.last_state_terminates() {
                ep_count += 1;
            }
        }
    }

    pub fn take_rollout_buffer(&mut self) -> RolloutBuffer<E::Tensor> {
        if !self.buffer.last_state_terminates() {
            let last_state = self.buffer.next_states.last().cloned();
            self.last_state = last_state;
        }
        self.buffer.take_rollout_buffer()
    }

    pub fn set_distr(&mut self, distr: Option<Box<dyn Distribution<Tensor = E::Tensor>>>) {
        self.distr = distr;
    }
}
