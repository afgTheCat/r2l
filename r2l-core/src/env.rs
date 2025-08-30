use crate::{distributions::Distribution, numeric::Buffer, utils::rollout_buffer::RolloutBuffer};
use bincode::{Decode, Encode};
use candle_core::Result;

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
pub struct SnapShot<T> {
    pub state: T,
    pub reward: f32,
    pub terminated: bool,
    pub trancuated: bool,
}

pub struct SnapShot2<T> {
    pub state: T,
    pub next_state: T,
    pub action: T,
    pub reward: f32,
    pub terminated: bool,
    pub trancuated: bool,
}

pub trait Env {
    // TODO: we need to figure the right tensor kind here!
    type Tensor: Clone;

    fn reset(&mut self, seed: u64) -> Self::Tensor;
    fn step(&mut self, action: Self::Tensor) -> SnapShot<Self::Tensor>;
    fn env_description(&self) -> EnvironmentDescription;
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}

// ok so the thing is this:
pub trait Sampler {
    type Env: Env;

    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
    ) -> Result<Vec<RolloutBuffer<D::Tensor>>>
    where
        <Self::Env as Env>::Tensor: From<D::Tensor>,
        <Self::Env as Env>::Tensor: Into<D::Tensor>;
}

pub type TensorOfSampler<S> = <<S as Sampler>::Env as Env>::Tensor;
