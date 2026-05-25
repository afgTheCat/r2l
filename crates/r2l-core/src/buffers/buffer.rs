use crate::{
    buffers::{
        Memory,
        // reusable_vec::{ReusableVec, ReusableVecSlice},
    },
    tensor::R2lTensor,
};

// the new buffer type I am experimenting with. Probably going to make things faster
#[derive(Clone)]
pub struct NewBuffer<T: R2lTensor> {
    states: Vec<T>,
    next_states: Vec<T>,
    actions: Vec<T>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl<T: R2lTensor> Default for NewBuffer<T> {
    fn default() -> Self {
        Self {
            states: Default::default(),
            next_states: Default::default(),
            actions: Default::default(),
            rewards: Default::default(),
            terminated: Default::default(),
            truncated: Default::default(),
        }
    }
}

pub struct TrajectoryView<'a, T: R2lTensor> {
    pub states: &'a [T],
    pub next_states: &'a [T],
    pub actions: &'a [T],
    pub rewards: &'a [f32],
    pub terminated: &'a [bool],
    pub truncated: &'a [bool],
}

impl<'a, T: R2lTensor> TrajectoryView<'a, T> {
    pub fn dones(&self) -> impl Iterator<Item = bool> {
        self.terminated
            .iter()
            .zip(self.truncated.iter())
            .map(|(terminated, truncated)| *terminated || *truncated)
    }

    pub fn episode_terminations(&self) -> usize {
        self.dones().filter(|x| *x).count()
    }
}

impl<T: R2lTensor> NewBuffer<T> {
    pub fn clear(&mut self) {
        self.states.clear();
        self.next_states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
    }

    pub fn push(&mut self, memory: Memory<T>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        } = memory;
        self.states.push(state);
        self.next_states.push(next_state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.truncated.push(truncated);
    }

    pub fn to_trajectory_view(&self) -> TrajectoryView<'_, T> {
        TrajectoryView {
            states: &self.states,
            next_states: &self.next_states,
            actions: &self.actions,
            rewards: &self.rewards,
            terminated: &self.terminated,
            truncated: &self.truncated,
        }
    }
}
