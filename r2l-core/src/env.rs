use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use anyhow::Result;
use bincode::{Decode, Encode};

#[derive(Debug, Clone)]
pub enum Space<T> {
    Discrete(usize),
    Continous {
        min: Option<T>,
        max: Option<T>,
        size: usize,
    },
}

impl<T> Space<T> {
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
pub struct EnvironmentDescription<T> {
    pub observation_space: Space<T>,
    pub action_space: Space<T>,
}

impl<T> EnvironmentDescription<T> {
    pub fn new(observation_space: Space<T>, action_space: Space<T>) -> Self {
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

pub trait Env {
    //  TODO: we might want to introduce more than just one kind of Tensors
    type Tensor: Clone + Send;

    fn reset(&mut self, seed: u64) -> Result<Self::Tensor>;
    fn step(&mut self, action: Self::Tensor) -> Result<SnapShot<Self::Tensor>>;
    fn env_description(&self) -> EnvironmentDescription<Self::Tensor>;
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
