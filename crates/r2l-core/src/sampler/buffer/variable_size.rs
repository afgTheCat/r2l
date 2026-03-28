use crate::{sampler::buffer::TrajectoryContainer, tensor::R2lTensor};

#[derive(Clone)]
pub struct VariableSizedStateBuffer<T: R2lTensor> {
    pub states: Vec<T>,
    pub next_states: Vec<T>,
    pub rewards: Vec<f32>,
    pub action: Vec<T>,
    pub terminated: Vec<bool>,
    pub trancuated: Vec<bool>,
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
