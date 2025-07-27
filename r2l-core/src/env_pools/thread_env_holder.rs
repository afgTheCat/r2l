use crate::env_pools::RolloutMode;
use crate::env_pools::{EnvHolder, SequentialVecEnvHooks};
use crate::rng::RNG;
use crate::{distributions::Distribution, env::Env, utils::rollout_buffer::RolloutBuffer};
use candle_core::{Error, Result, Tensor};
use crossbeam::channel::{Receiver, Sender};
use crossbeam::sync::ShardedLock;
use rand::Rng;
use std::sync::Arc;

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

pub struct WorkerThread<E: Env> {
    pub env: E,
    pub task_rx: Receiver<WorkerTask>,
    pub result_tx: Sender<ThreadResult>,
}

impl<E: Env> WorkerThread<E> {
    // signals whether a new job was found or not
    // thing is, we want a ReadWriteLock here, otherwise
    pub fn work(&mut self, distr: Arc<ShardedLock<Option<&dyn Distribution>>>) {
        while let Ok(task) = self.task_rx.recv() {
            match task {
                WorkerTask::Rollout {
                    rollout_mode,
                    mut initial_state,
                    env_idx,
                } => {
                    let Some(distr) = *distr.read().unwrap() else {
                        continue;
                    };
                    let mut state = initial_state.take().unwrap_or_else(|| {
                        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
                        self.env.reset(seed).unwrap()
                    });
                    // run_rollout_without_buffer(distr, &mut self.env, rollout_mode, &mut state)
                    //     .unwrap();
                    // let thread_result = ThreadResult {
                    //     states
                    // };
                    // self.result_tx.send(self.buff.clone()).unwrap();
                    todo!()
                }
                WorkerTask::SingleStep {
                    initial_state,
                    env_idx,
                } => todo!(),
                WorkerTask::Shutdown => continue,
            }
        }
    }
}

pub struct ThreadResult {
    states: Vec<(Tensor, Tensor, f32, bool, f32)>,
    last_state: Option<Tensor>,
    env_idx: usize, // TODO: this we probably don't need
}

pub struct ThreadHolder {
    pub worker_txs: Vec<Sender<WorkerTask>>,
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
    fn num_envs(&self) -> usize {
        self.worker_txs.len()
    }

    // TODO: Finish this!
    fn sequential_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<RolloutBuffer>> {
        self.lock_distr(distr);
        match rollout_mode {
            RolloutMode::StepBound { n_steps } => {
                todo!()
            }
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
        let num_envs = self.worker_txs.len();
        for env_idx in 0..num_envs {
            let tx = &mut self.worker_txs[env_idx];
            let initial_state = self.buffs[env_idx].last_state.take();
            let task = WorkerTask::Rollout {
                rollout_mode,
                initial_state,
                env_idx,
            };
            tx.send(task.clone()).unwrap();
        }
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
