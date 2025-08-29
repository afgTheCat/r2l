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
    fn to_rollout_buffer<D: Clone>(&mut self) -> RolloutBuffer<D>
    where
        E::Tensor: From<D>,
        E::Tensor: Into<D>,
    {
        let mut rb = RolloutBuffer::default();
        rb.states = std::mem::take(&mut self.states)
            .into_iter()
            .map(|b| b.into())
            .collect();
        rb.actions = std::mem::take(&mut self.action)
            .into_iter()
            .map(|b| b.into())
            .collect();
        rb.rewards = std::mem::take(&mut self.rewards);
        rb.dones = std::mem::take(&mut self.terminated)
            .into_iter()
            .zip(std::mem::take(&mut self.trancuated))
            .map(|(terminated, trancuated)| terminated || trancuated)
            .collect();
        let mut next_states = std::mem::take(&mut self.next_states);
        let last_state = next_states.pop().map(|b| b.into()).unwrap();
        rb.states.push(last_state.clone());
        rb
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
    pub last_state: Option<E::Tensor>,
}

impl<E: Env> VariableSizedTrajectoryBuffer<E> {
    pub fn new(env: E) -> Self {
        let buffer: VariableSizedStateBuffer<E> = VariableSizedStateBuffer::default();
        Self {
            env,
            buffer,
            last_state: None,
        }
    }

    pub fn step<DT: Clone, D: Distribution<Tensor = DT> + ?Sized>(&mut self, distr: &D)
    where
        E::Tensor: From<DT>,
        E::Tensor: Into<DT>,
    {
        let buffer = &mut self.buffer;
        let state = if let Some(obs) = buffer.next_states.last() {
            obs.clone()
        } else if let Some(last_state) = self.last_state.take() {
            last_state
        } else {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            self.env.reset(seed)
        };
        // TODO: we used to unsqueze here. I am guessing that we will have to provide a unified
        // structure to states and actions. Best way seem to be unsquezing 1d vectors
        let action = distr.get_action(state.clone().into()).unwrap();
        let SnapShot {
            state: mut next_state,
            reward,
            terminated,
            trancuated,
        } = self.env.step(action.clone().into());
        let done = terminated || trancuated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            next_state = self.env.reset(seed);
        }
        buffer.push(
            state,
            next_state,
            action.into(),
            reward,
            terminated,
            trancuated,
        );
    }

    pub fn step_with_epiosde_bound<DT: Clone, D: Distribution<Tensor = DT> + ?Sized>(
        &mut self,
        distr: &D,
        n_steps: usize,
    ) where
        E::Tensor: From<DT>,
        E::Tensor: Into<DT>,
    {
        let mut steps_taken = 0;
        loop {
            self.step(distr);
            steps_taken += 1;
            if steps_taken >= n_steps && self.buffer.last_state_terminates() {
                break;
            }
        }
    }

    pub fn run_episodes<DT: Clone, D: Distribution<Tensor = DT> + ?Sized>(
        &mut self,
        distr: &D,
        episodes: usize,
    ) where
        E::Tensor: From<DT>,
        E::Tensor: Into<DT>,
    {
        let mut ep_count = 0;
        while ep_count < episodes {
            self.step(distr);
            if self.buffer.last_state_terminates() {
                ep_count += 1;
            }
        }
    }

    pub fn to_rollout_buffer<D: Clone>(&mut self) -> RolloutBuffer<D>
    where
        E::Tensor: From<D>,
        E::Tensor: Into<D>,
    {
        if !self.buffer.last_state_terminates() {
            let last_state = self.buffer.next_states.last().cloned();
            self.last_state = last_state;
        }
        self.buffer.to_rollout_buffer()
    }
}
