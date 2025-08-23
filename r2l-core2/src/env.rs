use crate::{distributions::Distribution, numeric::Buffer};
use burn::{prelude::Backend, tensor::Tensor};

#[derive(Debug, Clone)]
pub enum Space {
    Discrete(usize),
    Continous {
        min: Option<Buffer>,
        max: Option<Buffer>,
        size: usize,
    },
}

impl Space {
    pub fn continous_from_dims(dims: Vec<usize>) -> Self {
        Self::Continous {
            min: None,
            max: None,
            size: dims.iter().product(),
        }
    }

    pub fn size(&self) -> usize {
        match &self {
            Self::Discrete(size) => *size,
            Self::Continous { size, .. } => *size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnvironmentDescription {
    pub observation_space: Space,
    pub action_space: Space,
}

impl EnvironmentDescription {
    pub fn new(observation_space: Space, action_space: Space) -> Self {
        Self {
            observation_space,
            action_space,
        }
    }

    pub fn action_size(&self) -> usize {
        self.action_space.size()
    }

    pub fn observation_size(&self) -> usize {
        self.observation_space.size()
    }
}

// TODO: This is a useful thing buffer needs to be go
pub struct SnapShot<Obs: Observation, Act: Action> {
    pub state: Obs,
    pub next_state: Obs,
    pub action: Act,
    pub reward: f32,
    pub terminated: bool,
    pub trancuated: bool,
}

impl<Obs: Observation, Act: Action> SnapShot<Obs, Act> {
    pub fn done(&self) -> bool {
        self.terminated || self.trancuated
    }

    pub fn reward(&self) -> f32 {
        self.reward
    }
}

// we probably also want some base types here since to to_tensor is and clamp is
pub trait Observation: Clone {
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1>;

    fn from_tensor<B: Backend>(t: Tensor<B, 1>) -> Self;
}

impl<B: Backend> Observation for Tensor<B, 2> {
    fn to_tensor<B2: Backend>(&self) -> Tensor<B2, 1> {
        todo!()
    }

    fn from_tensor<B2: Backend>(t: Tensor<B2, 1>) -> Self {
        todo!()
    }
}

pub trait Action: Clone {
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1>;

    fn from_tensor<B: Backend>(t: Tensor<B, 1>) -> Self;
}

impl<B: Backend> Action for Tensor<B, 2> {
    fn to_tensor<B2: Backend>(&self) -> Tensor<B2, 1> {
        todo!()
    }

    fn from_tensor<B2: Backend>(t: Tensor<B2, 1>) -> Self {
        todo!()
    }
}

pub trait Logp: Clone {
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1>;

    fn from_tensor<B: Backend>(t: Tensor<B, 1>) -> Self;
}

impl<B: Backend> Logp for Tensor<B, 2> {
    fn to_tensor<B2: Backend>(&self) -> Tensor<B2, 1> {
        todo!()
    }

    fn from_tensor<B2: Backend>(t: Tensor<B2, 1>) -> Self {
        todo!()
    }
}

pub trait Env {
    type Obs: Observation;
    type Act: Action;

    fn reset(&mut self, seed: u64) -> Self::Obs;
    fn step(&mut self, action: &Self::Act) -> SnapShot<Self::Obs, Self::Act>;
    fn env_description(&self) -> EnvironmentDescription;
}

// This is mostly internal to r2l
pub trait Actor {
    type Obs: Observation;
    type Act: Action;

    fn collect_rollouts<O: Observation, A: Action, D: Distribution>(
        &mut self,
        distribution: D,
    ) -> Vec<SnapShot<Self::Obs, Self::Act>>
    where
        O: From<Self::Obs>,
        A: From<Self::Act>,
        Self::Obs: From<O>,
        Self::Act: From<A>;
}
