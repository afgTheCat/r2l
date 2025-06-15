pub mod dummy_vec_env;
pub mod sub_processing_vec_env;
pub mod vec_env;

use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use bincode::{Decode, Encode};
use candle_core::{Result, Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::cell::RefCell;

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}

pub trait Env {
    fn reset(&self, seed: u64) -> Result<Tensor>;
    fn step(&self, action: &Tensor) -> Result<(Tensor, f32, bool, bool)>;
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub enum RolloutMode {
    EpisodeBound { n_steps: usize },
    StepBound { n_steps: usize },
}

pub trait EnvPool {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>>;
}

fn single_step_env(
    dist: &impl Distribution,
    state: &Tensor,
    env: &impl Env,
) -> Result<(Tensor, Tensor, f32, bool, Tensor)> {
    // TODO: unsqueezing here is kinda ugly, we probably need the dist to enforce some shape
    // requirement
    let (action, logp) = dist.get_action(&state.unsqueeze(0)?)?;
    let (mut next_state, reward, terminated, trancuated) = env.step(&action)?;
    if terminated || trancuated {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        next_state = env.reset(seed)?;
    }
    Ok((next_state, action, reward, terminated || trancuated, logp))
}

pub fn run_rollout(
    dist: &impl Distribution,
    env: &impl Env,
    collection_type: RolloutMode,
    rollout_buffer: &mut RolloutBuffer,
) -> Result<()> {
    let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
    let mut state = rollout_buffer.reset(env, seed)?;
    match collection_type {
        RolloutMode::EpisodeBound { n_steps } => loop {
            let (next_state, action, reward, done, logp) = single_step_env(dist, &state, env)?;
            let logp = logp.to_scalar()?;
            rollout_buffer.push_step(state.clone(), action, reward, done, logp);
            state = next_state;
            if rollout_buffer.states.len() >= n_steps && done {
                break;
            }
        },
        RolloutMode::StepBound { n_steps } => {
            for _ in 0..n_steps {
                let (next_state, action, reward, done, logp) = single_step_env(dist, &state, env)?;
                let logp = logp.squeeze(0)?.to_scalar()?;
                rollout_buffer.push_step(state, action, reward, done, logp);
                state = next_state;
            }
        }
    }
    rollout_buffer.push_state(state);
    Ok(())
}
