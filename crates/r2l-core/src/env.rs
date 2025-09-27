use std::fmt::Debug;

use crate::{
    distributions::Policy, sampler2::Buffer, sampler3::buffers::BufferStack, tensor::R2lTensor,
    utils::rollout_buffer::RolloutBuffer,
};
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

pub struct Memory<T> {
    pub state: T,
    pub next_state: T,
    pub action: T,
    pub reward: f32,
    pub terminated: bool,
    pub trancuated: bool,
}

pub trait Env {
    //  TODO: we might want to introduce more than just one kind of Tensors.
    type Tensor: R2lTensor;

    fn reset(&mut self, seed: u64) -> Result<Self::Tensor>;
    fn step(&mut self, action: Self::Tensor) -> Result<SnapShot<Self::Tensor>>;
    fn env_description(&self) -> EnvironmentDescription<Self::Tensor>;
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}

// TODO:
pub trait Sampler {
    type Env: Env;

    fn collect_rollouts<D: Policy + Clone>(
        &mut self,
        distribution: D,
    ) -> Result<Vec<RolloutBuffer<D::Tensor>>>
    where
        <Self::Env as Env>::Tensor: From<D::Tensor>,
        <Self::Env as Env>::Tensor: Into<D::Tensor>;
}

pub type TensorOfSampler<S> = <<S as Sampler>::Env as Env>::Tensor;

// TODO: we don't really want a Vec of buffers exposed. In the grand scheme of things, the agent
// should not really care for how the Buffer is collected. It should only care about the. What we
// want is basically a MemorySampler that has the same Tensor type what the agent has. That means
// that a new trait should probably be exposed. Some built in features would be nice as well.
pub trait Sampler2 {
    type E: Env;
    type Buffer: Buffer<Tensor = <Self::E as Env>::Tensor>;

    fn collect_rollouts<P: Policy + Clone>(&mut self, policy: P) -> Result<Vec<Self::Buffer>>
    where
        <Self::Buffer as Buffer>::Tensor: From<P::Tensor>,
        <Self::Buffer as Buffer>::Tensor: Into<P::Tensor>;
}

pub type TensorOfSampler2<S> = <<S as Sampler2>::Buffer as Buffer>::Tensor;

pub trait EnvBuilderTrait: Sync + Send + 'static {
    type Env: Env;

    fn build_env(&self) -> Result<Self::Env>;
}

impl<E: Env, F: Sync + Send + 'static> EnvBuilderTrait for F
where
    F: Fn() -> Result<E>,
{
    type Env = E;

    fn build_env(&self) -> Result<Self::Env> {
        (self)()
    }
}

pub trait Sampler3 {
    type E: Env;
    type Buffer: Buffer<Tensor = <Self::E as Env>::Tensor>;

    fn collect_rollouts<P: Policy + Clone>(&mut self, policy: P) -> BufferStack<Self::Buffer>
    where
        <Self::Buffer as Buffer>::Tensor: From<P::Tensor>,
        <Self::Buffer as Buffer>::Tensor: Into<P::Tensor>;
}

pub type TensorOfSampler3<S> = <<S as Sampler3>::Buffer as Buffer>::Tensor;
