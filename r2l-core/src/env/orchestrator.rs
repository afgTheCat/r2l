use candle_core::{Error, Result, Tensor};
use crossbeam::channel::{Receiver, Sender};
use crossbeam::sync::ShardedLock;
use enum_dispatch::enum_dispatch;
use std::sync::Arc;

use crate::env::{EnvironmentDescription, RolloutMode, run_rollout_without_buffer};
use crate::{
    distributions::Distribution,
    env::{Env, sequential_vec_env::SequentialVecEnvHooks, single_step_env},
    utils::rollout_buffer::RolloutBuffer,
};

#[enum_dispatch]
pub trait EnvHolder {
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

pub struct VecEnvHolder<E: Env> {
    pub envs: Vec<E>,
    pub buffers: Vec<RolloutBuffer>,
}

impl<E: Env> VecEnvHolder<E> {
    fn single_step_env(
        &mut self,
        distr: &impl Distribution,
        current_states: &mut Vec<Tensor>,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<(Tensor, Tensor, f32, f32, bool)>> {
        let mut states = self
            .envs
            .iter_mut()
            .zip(current_states.iter())
            .map(|(env, state)| single_step_env(distr, state, env))
            .collect::<Result<Vec<_>>>()?;
        // TODO: this is kinda jank like this. Maybe we need the previous state here?
        hooks.step_hook(distr, &mut states)?;
        // save the states. This is super not obvious what is going on here
        for (idx, rb) in self.buffers.iter_mut().enumerate() {
            let state = &current_states[idx];
            let (next_state, action, reward, logp, done) = &states[idx];
            rb.push_step(state.clone(), action.clone(), *reward, *done, *logp);
            current_states[idx] = next_state.clone();
        }
        Ok(states)
    }
}

impl<E: Env> EnvHolder for VecEnvHolder<E> {
    fn sequential_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<RolloutBuffer>> {
        let mut current_states = self
            .buffers
            .iter_mut()
            .enumerate()
            .map(|(env_idx, rb)| rb.reset(&mut self.envs[env_idx], rand::random()))
            .collect::<Result<Vec<_>>>()?;
        let mut steps_taken = 0;
        let num_envs = self.envs.len();
        match rollout_mode {
            RolloutMode::StepBound { n_steps } => {
                while steps_taken < n_steps {
                    self.single_step_env(distr, &mut current_states, hooks)?;
                    steps_taken += num_envs;
                }
                // TODO: what should this be? The current state?
                hooks.post_step_hook(&mut current_states)?;
                for (rb_idx, rb) in self.buffers.iter_mut().enumerate() {
                    rb.set_last_state(current_states[rb_idx].clone());
                }
            }
            RolloutMode::EpisodeBound { n_episodes } => {
                todo!()
            }
        }
        Ok(self.buffers.clone())
    }

    // not really async in this case, only each environment does the thing after one another
    fn async_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        for (env_idx, buffer) in self.buffers.iter_mut().enumerate() {
            let env = &mut self.envs[env_idx];
            let mut current_state = buffer.reset(env, rand::random())?;
            let states = run_rollout_without_buffer(distr, env, rollout_mode, &mut current_state)?;
            buffer.set_states(states, current_state);
        }
        Ok(self.buffers.clone())
    }
}

#[derive(Clone)]
pub enum WorkerTask {
    Rollout {
        rollout_mode: RolloutMode,
        initial_state: Option<Tensor>,
        env_idx: usize,
    },
    SingleStep {
        initial_state: Tensor,
        env_idx: usize,
    },
    Shutdown,
}

pub struct ThreadResult {
    states: Vec<(Tensor, Tensor, f32, f32, bool)>,
    last_state: Option<Tensor>,
    env_idx: usize, // TODO: this we probably don't need
}

pub struct ThreadHolder {
    pub worker_txs: Vec<Sender<WorkerTask>>,
    // TODO: should not be a rollout buffer
    pub result_rx: Receiver<ThreadResult>,
    pub distr_lock: Arc<ShardedLock<Option<&'static dyn Distribution>>>,
    pub buffs: Vec<RolloutBuffer>,
    pub current_states: Vec<Tensor>,
}

impl ThreadHolder {
    fn lock_distr<D: Distribution>(&mut self, distr: &D) {
        // SAFETY:
        // We cast `&D` to a `'static` lifetime in order to temporarily store it in a sharded lock.
        // This is sound because access to the distribution is strictly synchronized through a
        // lock. Each lock `lock_distr` has to follow with an `un_lock_distr`.
        let distr_ref: &'static D = unsafe { std::mem::transmute(distr) };
        let mut distr_lock = self.distr_lock.write().unwrap();
        *distr_lock = Some(distr_ref);
    }

    fn unlock_distr(&mut self) {
        let mut distr_lock = self.distr_lock.write().unwrap();
        *distr_lock = None
    }

    fn single_step_env(&mut self) -> Result<()> {
        // send the states
        for env_idx in 0..self.buffs.len() {
            let state = self.current_states[env_idx].clone();
            self.worker_txs[env_idx]
                .send(WorkerTask::SingleStep {
                    initial_state: state,
                    env_idx,
                })
                .map_err(|err| Error::wrap(err))?;
        }

        // receive the states
        for _ in 0..self.buffs.len() {
            let ThreadResult {
                states,
                last_state,
                env_idx,
            } = self.result_rx.recv().map_err(|err| Error::wrap(err))?;
            debug_assert!(states.len() == 1);
            todo!()
        }

        Ok(())
    }
}

impl EnvHolder for ThreadHolder {
    fn sequential_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<RolloutBuffer>> {
        self.lock_distr(distr);
        match rollout_mode {
            RolloutMode::StepBound { n_steps } => {}
            RolloutMode::EpisodeBound { n_episodes } => {
                todo!()
            }
        }

        self.unlock_distr();
        Ok(self.buffs.clone())
    }

    fn async_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        self.lock_distr(distr);
        for env_idx in 0..self.buffs.len() {
            let tx = &mut self.worker_txs[env_idx];
            let initial_state = self.buffs[env_idx].last_state.take();
            let task = WorkerTask::Rollout {
                rollout_mode,
                initial_state,
                env_idx,
            };
            tx.send(task.clone()).unwrap();
        }
        let num_envs = self.worker_txs.len();
        for _ in 0..num_envs {
            let ThreadResult {
                states,
                last_state,
                env_idx,
            } = self.result_rx.recv().map_err(|err| Error::wrap(err))?;
            let last_state = last_state.unwrap();
            let buffer = &mut self.buffs[env_idx];
            buffer.set_states(states, last_state);
        }
        self.unlock_distr();
        Ok(self.buffs.clone())
    }
}

pub struct SubprocHolder {}

impl EnvHolder for SubprocHolder {
    fn sequential_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }

    fn async_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }
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

impl<H: EnvHolder> R2lEnvPool<H> {
    pub fn step(
        &mut self,
        distr: &impl Distribution,
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
}
