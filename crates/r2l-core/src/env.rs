use std::fmt::Debug;

use crate::{
    distributions::Policy,
    sampler3::{
        buffer_stack::BufferStack3,
        buffers::{Buffer, BufferStack},
    },
    tensor::R2lTensor,
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

pub trait Sampler3 {
    type E: Env;
    type Buffer: Buffer<Tensor = <Self::E as Env>::Tensor>;

    // TODO: what if we return the BufferStack according to the policies state vec?
    fn collect_rollouts<P: Policy + Clone>(&mut self, policy: P) -> BufferStack<Self::Buffer>
    where
        <Self::Buffer as Buffer>::Tensor: From<P::Tensor>,
        <Self::Buffer as Buffer>::Tensor: Into<P::Tensor>;
}

pub type TensorOfSampler3<S> = <<S as Sampler3>::Buffer as Buffer>::Tensor;

// TODO: I think this is the final form! Also this is the best of the best. What we need is
// basically a structure that converts things! This way we can also get rid of the Buffer trait
// but we can have the BufferStack exposed. Tbh BufferStack is the innovation here, not the
// buffers. Also we might want the Buffer trait kept.
pub trait Sampler4 {
    type Env: Env;

    fn collect_rollouts<P: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        policy: P,
    );

    fn get_buffer_stack<T: R2lTensor + From<<Self::Env as Env>::Tensor>>(&self) -> BufferStack3<T>;
}
