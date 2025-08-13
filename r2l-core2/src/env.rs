use crate::{distributions::Distribution, numeric::Buffer};
use bincode::{Decode, Encode};
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
pub struct SnapShot<Obs: Observation> {
    pub state: Obs,
    pub reward: f32,
    pub terminated: bool,
    pub trancuated: bool,
}

// we probably also want some base types here since to to_tensor is and clamp is
pub trait Observation {
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1>;

    fn from_tensor<B: Backend>(t: Tensor<B, 1>) -> Self;
}

pub trait Action {
    fn to_tensor<B: Backend>(&self) -> Tensor<B, 1>;

    fn from_tensor<B: Backend>(t: Tensor<B, 1>) -> Self;
}

pub trait Env {
    type Obs: Observation;
    type Act: Action;

    // reset returns an observation, which should be an associated type, not the buffer itself
    fn reset(&mut self, seed: u64) -> Self::Obs;
    fn step(&mut self, action: &Self::Act) -> SnapShot<Self::Obs>;
    fn env_description(&self) -> EnvironmentDescription;
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}

// This is mostly internal to r2l
pub trait EnvPool {
    fn collect_rollouts<O: Observation, A: Action, D: Distribution<O, A>>(
        &mut self,
        distribution: D,
        rollout_mode: RolloutMode,
    ) -> Vec<SnapShot<O>>;

    fn env_description(&self) -> EnvironmentDescription;

    fn num_env(&self) -> usize;
}
