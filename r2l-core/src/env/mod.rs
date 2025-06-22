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

pub fn single_step_env(
    dist: &impl Distribution,
    state: &Tensor,
    env: &impl Env,
) -> Result<(Tensor, Tensor, f32, f32, bool)> {
    // TODO: unsqueezing here is kinda ugly, we probably need the dist to enforce some shape
    let (action, logp) = dist.get_action(&state.unsqueeze(0)?)?;
    let (mut next_state, reward, terminated, trancuated) = env.step(&action)?;
    let done = terminated || trancuated;
    if done {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        next_state = env.reset(seed)?;
    }
    let logp: f32 = logp.squeeze(0)?.to_scalar()?;
    Ok((next_state, action, reward, logp, done))
}

pub fn single_step_env_with_buffer(
    dist: &impl Distribution,
    state: &Tensor,
    env: &impl Env,
    rollout_buffer: &mut RolloutBuffer,
) -> Result<(Tensor, bool)> {
    let (next_state, action, reward, logp, done) = single_step_env(dist, state, env)?;
    rollout_buffer.push_step(state.clone(), action, reward, done, logp);
    Ok((next_state, done))
}

pub trait StepHook<D: Distribution>: Fn(&D) -> Result<()> {}

pub fn run_rollout<D: Distribution>(
    dist: &D,
    env: &impl Env,
    rollout_mode: RolloutMode,
    rollout_buffer: &mut RolloutBuffer,
    step_hook: Option<Box<dyn StepHook<D>>>,
) -> Result<()> {
    let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
    let mut state = rollout_buffer.reset(env, seed)?;
    match rollout_mode {
        RolloutMode::EpisodeBound { n_steps } => loop {
            let (next_state, done) =
                single_step_env_with_buffer(dist, &state, env, rollout_buffer)?;
            if let Some(step_hook) = &step_hook {
                step_hook(dist)?;
            };
            state = next_state;
            if rollout_buffer.states.len() >= n_steps && done {
                break;
            }
        },
        RolloutMode::StepBound { n_steps } => {
            for _ in 0..n_steps {
                let (next_state, _done) =
                    single_step_env_with_buffer(dist, &state, env, rollout_buffer)?;
                if let Some(step_hook) = &step_hook {
                    step_hook(dist)?;
                };
                state = next_state;
            }
        }
    }
    rollout_buffer.set_last_state(state);
    Ok(())
}
