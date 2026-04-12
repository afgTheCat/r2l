use crate::{
    buffers::{ExpandableTrajectoryContainer, Memory, TrajectoryContainer},
    tensor::R2lTensor,
};

#[derive(Clone)]
pub struct VariableSizedStateBuffer<T: R2lTensor> {
    pub states: Vec<T>,
    pub next_states: Vec<T>,
    pub rewards: Vec<f32>,
    pub action: Vec<T>,
    pub terminated: Vec<bool>,
    pub trancuated: Vec<bool>,
}

impl<T: R2lTensor> Default for VariableSizedStateBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: R2lTensor> VariableSizedStateBuffer<T> {
    pub fn new() -> Self {
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

impl<T: R2lTensor> TrajectoryContainer for VariableSizedStateBuffer<T> {
    type Tensor = T;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn states(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.states.iter()
    }

    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.next_states.iter()
    }

    fn actions(&self) -> impl Iterator<Item = &Self::Tensor> {
        self.action.iter()
    }

    fn rewards(&self) -> impl Iterator<Item = f32> {
        self.rewards.iter().copied()
    }

    fn terminated(&self) -> impl Iterator<Item = bool> {
        self.terminated.iter().copied()
    }

    fn trancuated(&self) -> impl Iterator<Item = bool> {
        self.trancuated.iter().copied()
    }
}

impl<T: R2lTensor> ExpandableTrajectoryContainer for VariableSizedStateBuffer<T> {
    fn push(&mut self, memory: Memory<Self::Tensor>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        } = memory;
        self.states.push(state);
        self.next_states.push(next_state);
        self.action.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.trancuated.push(trancuated);
    }
}
