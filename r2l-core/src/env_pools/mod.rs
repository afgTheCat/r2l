pub mod subproc_env_holder;
pub mod thread_env_holder;
pub mod vector_env_holder;

use crate::{
    distributions::Distribution,
    env::{Env, EnvPool, EnvironmentDescription, RolloutMode},
    env_pools::{
        subproc_env_holder::SubprocHolder, thread_env_holder::ThreadHolder,
        vector_env_holder::VecEnvHolder,
    },
    numeric::Buffer,
    rng::RNG,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Device, Result, Tensor};
use enum_dispatch::enum_dispatch;
use rand::Rng;

pub trait SequentialVecEnvHooks {
    fn step_hook(
        &mut self,
        distribution: &dyn Distribution,
        states: &mut Vec<(Tensor, Tensor, f32, f32, bool)>,
    ) -> candle_core::Result<bool>;

    fn post_step_hook(&mut self, last_states: &mut Vec<Tensor>) -> candle_core::Result<bool>;
}

#[enum_dispatch]
pub trait EnvHolder {
    fn num_envs(&self) -> usize;

    fn sequential_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<RolloutBuffer>>;

    fn async_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>>;
}

pub fn single_step_env(
    distr: &dyn Distribution,
    state: &Tensor,
    env: &mut impl Env,
    device: &Device,
) -> Result<(Tensor, Tensor, f32, f32, bool)> {
    // TODO: unsqueezing here is kinda ugly, we probably need the dist to enforce some shape
    let (action, logp) = distr.get_action(&state.unsqueeze(0)?)?;
    let (next_state, reward, terminated, trancuated) =
        env.step(&Buffer::from_candle_tensor(&action));
    let mut next_state = next_state.to_candle_tensor(device);
    let done = terminated || trancuated;
    if done {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        next_state = env.reset(seed).to_candle_tensor(device);
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
    device: &Device,
) -> Result<(Tensor, bool)> {
    let (next_state, action, reward, logp, done) = single_step_env(dist, state, env, device)?;
    rollout_buffer.push_step(state.clone(), action, reward, done, logp);
    Ok((next_state, done))
}

pub fn run_rollout(
    distr: &dyn Distribution,
    env: &mut impl Env,
    rollout_mode: RolloutMode,
    mut state: Tensor,
    device: &Device,
) -> Result<(Vec<(Tensor, Tensor, f32, bool, f32)>, Tensor)> {
    let mut res = vec![];
    match rollout_mode {
        RolloutMode::EpisodeBound { n_episodes } => {
            // just a loop
            loop {
                let (next_state, action, reward, logp, done) =
                    single_step_env(distr, &state, env, device)?;
                res.push((state.clone(), action, reward, done, logp));
                state = next_state;
                if res.len() >= n_episodes && done {
                    break;
                }
            }
        }
        RolloutMode::StepBound { n_steps } => {
            for _ in 0..n_steps {
                let (next_state, action, reward, logp, done) =
                    single_step_env(distr, &state, env, device)?;
                res.push((state.clone(), action, reward, done, logp));
                // buffer.push_step(state.clone(), action, reward, done, logp);
                state = next_state;
            }
        }
    }
    Ok((res, state))
}

#[enum_dispatch(EnvHolder)]
pub enum R2lEnvHolder<E: Env> {
    Vec(VecEnvHolder<E>),
    Thread(ThreadHolder),
    SubProc(SubprocHolder),
}

pub enum StepMode {
    Sequential(Box<dyn SequentialVecEnvHooks>),
    Async,
}

// No generics should be here, but also I should be using a triat. We also want this to be part of
// the core thing.
pub struct R2lEnvPool<H: EnvHolder> {
    pub env_holder: H,
    pub step_mode: StepMode,
    pub env_description: EnvironmentDescription,
}

impl<H: EnvHolder> EnvPool for R2lEnvPool<H> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        match &mut self.step_mode {
            StepMode::Sequential(hooks) => {
                self.env_holder
                    .sequential_rollout(distr, rollout_mode, hooks.as_mut())
            }
            StepMode::Async => self.env_holder.async_rollout(distr, rollout_mode),
        }
    }

    fn env_description(&self) -> EnvironmentDescription {
        self.env_description.clone()
    }

    fn num_env(&self) -> usize {
        self.env_holder.num_envs()
    }
}
