pub mod dummy_vec_env;
pub mod orchestrator;
pub mod sequential_vec_env;
pub mod sub_processing_vec_env;
pub mod vec_env;

use crate::{
    distributions::Distribution,
    env::{
        dummy_vec_env::DummyVecEnv, sequential_vec_env::SequentialVecEnv,
        sub_processing_vec_env::SubprocessingEnv, vec_env::VecEnv,
    },
    rng::RNG,
    utils::rollout_buffer::RolloutBuffer,
};
use bincode::{Decode, Encode};
use candle_core::{Result, Tensor};
use rand::Rng;

#[derive(Debug, Clone)]
pub enum Space {
    Discrete(usize),
    Continous {
        min: Option<Tensor>,
        max: Option<Tensor>,
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

pub trait Env {
    fn reset(&mut self, seed: u64) -> Result<Tensor>;
    fn step(&mut self, action: &Tensor) -> Result<(Tensor, f32, bool, bool)>;
    // TODO: do we need this?
    fn env_description(&self) -> EnvironmentDescription;
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
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

pub trait EnvPool {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>>;

    fn env_description(&self) -> EnvironmentDescription;

    fn num_env(&self) -> usize;
}

pub fn single_step_env(
    distr: &dyn Distribution,
    state: &Tensor,
    env: &mut impl Env,
) -> Result<(Tensor, Tensor, f32, f32, bool)> {
    // TODO: unsqueezing here is kinda ugly, we probably need the dist to enforce some shape
    let (action, logp) = distr.get_action(&state.unsqueeze(0)?)?;
    let (mut next_state, reward, terminated, trancuated) = env.step(&action)?;
    let done = terminated || trancuated;
    if done {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        next_state = env.reset(seed)?;
    }
    let logp: f32 = logp.squeeze(0)?.to_scalar()?;
    Ok((next_state, action, reward, logp, done))
}

// TODO: retire this
pub fn single_step_env_with_buffer(
    dist: &dyn Distribution,
    state: &Tensor,
    env: &mut impl Env,
    rollout_buffer: &mut RolloutBuffer,
) -> Result<(Tensor, bool)> {
    let (next_state, action, reward, logp, done) = single_step_env(dist, state, env)?;
    rollout_buffer.push_step(state.clone(), action, reward, done, logp);
    Ok((next_state, done))
}

// TODO: retire this
pub fn run_rollout(
    dist: &dyn Distribution,
    env: &mut impl Env,
    rollout_mode: RolloutMode,
    rollout_buffer: &mut RolloutBuffer,
) -> Result<()> {
    let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
    let mut state = rollout_buffer.reset(env, seed)?;
    match rollout_mode {
        RolloutMode::EpisodeBound {
            n_episodes: n_steps,
        } => loop {
            let (next_state, done) =
                single_step_env_with_buffer(dist, &state, env, rollout_buffer)?;
            state = next_state;
            if rollout_buffer.states.len() >= n_steps && done {
                break;
            }
        },
        RolloutMode::StepBound { n_steps } => {
            for _ in 0..n_steps {
                let (next_state, _done) =
                    single_step_env_with_buffer(dist, &state, env, rollout_buffer)?;
                state = next_state;
            }
        }
    }
    rollout_buffer.set_last_state(state);
    Ok(())
}

pub fn run_rollout_without_buffer(
    distr: &dyn Distribution,
    env: &mut impl Env,
    rollout_mode: RolloutMode,
    state: &mut Tensor,
) -> Result<Vec<(Tensor, Tensor, f32, f32, bool)>> {
    let mut res = vec![];
    match rollout_mode {
        RolloutMode::EpisodeBound { n_episodes } => {
            // just a loop
            loop {
                let (next_state, action, reward, logp, done) = single_step_env(distr, &state, env)?;
                res.push((next_state.clone(), action, reward, logp, done));
                *state = next_state;
                if res.len() >= n_episodes && done {
                    break;
                }
            }
        }
        RolloutMode::StepBound { n_steps } => {
            for _ in 0..n_steps {
                let (next_state, action, reward, logp, done) = single_step_env(distr, &state, env)?;
                res.push((next_state.clone(), action, reward, logp, done));
                *state = next_state;
            }
        }
    }
    Ok(res)
}

// TODO: Restricting here is probably not neccessary
pub enum EnvPoolType<E: Env> {
    Dummy(DummyVecEnv<E>),
    VecEnv(VecEnv),
    Subprocessing(SubprocessingEnv),
    Sequential(SequentialVecEnv<E>),
}

impl<E: Env + Sync> EnvPool for EnvPoolType<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        match self {
            Self::Dummy(vec_env) => vec_env.collect_rollouts(distribution, rollout_mode),
            Self::Sequential(vec_env) => vec_env.collect_rollouts(distribution, rollout_mode),
            _ => todo!(),
        }
    }

    fn env_description(&self) -> EnvironmentDescription {
        match &self {
            Self::Dummy(vec_env) => vec_env.env_description(),
            Self::Sequential(vec_env) => vec_env.env_description(),
            _ => todo!(),
        }
    }

    fn num_env(&self) -> usize {
        match &self {
            Self::Dummy(vec_env) => vec_env.num_env(),
            Self::Sequential(vec_env) => vec_env.num_env(),
            _ => todo!(),
        }
    }
}
