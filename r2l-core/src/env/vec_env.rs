use super::RolloutMode;
use super::{Env, EnvPool};
use crate::env::{EnvironmentDescription, run_rollout};
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::{Error, Result};
use crossbeam::channel::{Receiver, Sender};
use crossbeam::sync::ShardedLock;
use std::sync::Arc;

#[derive(Clone)]
pub enum WorkerTask {
    Rollout(RolloutMode),
    Shutdown,
}

pub struct WorkerThread<E: Env> {
    pub env: E,
    pub buff: RolloutBuffer,
    pub task_rx: Receiver<WorkerTask>,
    pub result_tx: Sender<RolloutBuffer>,
}

impl<E: Env> WorkerThread<E> {
    // signals whether a new job was found or not
    // thing is, we want a ReadWriteLock here, otherwise
    pub fn work(&mut self, distr: Arc<ShardedLock<Option<&dyn Distribution>>>) {
        while let Ok(task) = self.task_rx.recv() {
            match task {
                WorkerTask::Rollout(rollout) => {
                    let Some(distr) = *distr.read().unwrap() else {
                        continue;
                    };
                    run_rollout(distr, &mut self.env, rollout, &mut self.buff).unwrap();
                    self.result_tx.send(self.buff.clone()).unwrap();
                }
                WorkerTask::Shutdown => continue,
            }
        }
    }
}

pub struct VecEnv {
    pub worker_txs: Vec<Sender<WorkerTask>>,
    pub result_rx: Receiver<RolloutBuffer>,
    pub env_description: EnvironmentDescription,
    pub distr_lock: Arc<ShardedLock<Option<&'static dyn Distribution>>>,
}

impl VecEnv {
    pub fn shutdown(&mut self) {
        for tx in self.worker_txs.iter_mut() {
            tx.send(WorkerTask::Shutdown).unwrap();
        }
    }
}

impl EnvPool for VecEnv {
    fn env_description(&self) -> EnvironmentDescription {
        self.env_description.clone()
    }

    fn num_env(&self) -> usize {
        self.worker_txs.len()
    }

    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        let mut distr_lock = self.distr_lock.write().unwrap();

        // SAFETY:
        // We cast `&D` to a `'static` lifetime in order to temporarily store it in a shared lock.
        //
        // This is sound because access to the distribution is strictly synchronized through a lock:
        // - The lock is a `ShardedLock<Option<&'static D>>`, initially `None`.
        // - We acquire a write lock to set it to `Some(distr_ref)`, which guarantees exclusive access.
        // - After all intended accesses are complete, we re-acquire the write lock and set it back to `None`.
        let distr_ref: &'static D = unsafe { std::mem::transmute(distribution) };
        *distr_lock = Some(distr_ref);
        drop(distr_lock);

        let task = WorkerTask::Rollout(rollout_mode);
        for tx in self.worker_txs.iter_mut() {
            tx.send(task.clone()).unwrap();
        }
        let num_envs = self.worker_txs.len();
        let rollouts = (0..num_envs)
            .map(|_| self.result_rx.recv().map_err(|err| Error::wrap(err)))
            .collect();

        let mut distr_lock = self.distr_lock.write().unwrap();
        *distr_lock = None;
        rollouts
    }
}
