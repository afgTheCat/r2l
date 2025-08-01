use crate::{distributions::Distribution, numeric::Buffer, utils::rollout_buffer::RolloutBuffer};
use bincode::{Decode, Encode};
use candle_core::{Result, Tensor};

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

pub trait Env {
    fn reset(&mut self, seed: u64) -> Buffer;
    fn step(&mut self, action: &Buffer) -> (Buffer, f32, bool, bool);
    fn env_description(&self) -> EnvironmentDescription;
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}

// TODO: we may want to get rid of the env pool trait and submerge it into the env trait
pub trait EnvPool {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>>;

    fn env_description(&self) -> EnvironmentDescription;

    fn num_env(&self) -> usize;
}
