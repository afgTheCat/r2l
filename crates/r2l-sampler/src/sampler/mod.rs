mod workers;

use std::sync::Arc;

use bimodal_array::{ArrayGuard, ArrayHandle, ArrayMap, ElementHandle, bimodal_array};
use crossbeam::channel::unbounded;
use r2l_core::buffers::buffer::TrajectoryView;
use r2l_core::buffers::{Memory, MultiMemory};
use r2l_core::env::normalizer::{Normalizer, NormalizerMode};
use r2l_core::env::{EnvBuilder, EnvBuilderType, Snapshot};
use r2l_core::error::{Error, Result};
use r2l_core::on_policy::algorithm::Sampler;
use r2l_core::rng::sample_u64;
use r2l_core::{buffers::buffer::TrajectoryBuffer, env::Env, models::Actor, tensor::R2lTensor};
use workers::{PoolWorker, ThreadWorker, Workers};

use crate::{RolloutMode, SamplerExecutionMode, SamplerHookResult};

struct Worker<E: Env> {
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<E: Env> Worker<E> {
    fn build(env_builder: &EnvBuilderType<E>, idx: usize) -> Result<(Self, E::Tensor)> {
        let mut env = env_builder.build_idx(idx)?;
        let state = env.reset(sample_u64())?;
        Ok((Self { env, actor: None }, state))
    }

    fn get_policy(&mut self) -> Result<&mut Box<dyn Actor<Tensor = E::Tensor>>> {
        self.actor.as_mut().ok_or_else(|| {
            Error::invalid_state("step sampler worker", "no policy has been installed")
        })
    }

    fn step(&mut self, state: &mut E::Tensor) -> Result<Memory<E::Tensor>> {
        let policy = self.get_policy()?;
        let current_state = state.clone();
        let action = policy.action(state.clone())?;
        let Snapshot {
            state: next_state,
            reward,
            terminated,
            truncated,
        } = self.env.step(action.clone())?;
        let done = terminated || truncated;
        *state = if done {
            self.env.reset(sample_u64())?
        } else {
            next_state.clone()
        };
        Ok(Memory {
            state: current_state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        })
    }
}

// Direct workers keep their state locally and write whole rollouts to a buffer.
struct LocalStateWorker<E: Env> {
    state: E::Tensor,
    worker: Worker<E>,
    buffer: ElementHandle<TrajectoryBuffer<E::Tensor>>,
}

impl<E: Env> PoolWorker for LocalStateWorker<E> {
    type Tensor = E::Tensor;
    type Request = RolloutMode;
    type Output = ();

    fn set_policy(&mut self, actor: Box<dyn Actor<Tensor = E::Tensor>>) {
        self.worker.actor = Some(actor);
    }

    fn reset(&mut self) -> Result<()> {
        self.state = self.worker.env.reset(sample_u64())?;
        Ok(())
    }

    fn collect(&mut self, bound: RolloutMode) -> Result<()> {
        let mut buffer = self.buffer.lock().unwrap();
        match bound {
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = 0;
                loop {
                    let memory = self.worker.step(&mut self.state)?;
                    if memory.is_done() {
                        episodes += 1;
                    }
                    buffer.push(memory);
                    if episodes >= n_episodes.get() {
                        break;
                    }
                }
            }
            RolloutMode::StepBound { n_steps } => {
                for _ in 0..n_steps.get() {
                    let memory = self.worker.step(&mut self.state)?;
                    buffer.push(memory);
                }
            }
        }
        Ok(())
    }
}

// The coordinator normalizes staged states between worker steps.
struct StagedWorker<E: Env> {
    state: ElementHandle<E::Tensor>,
    worker: Worker<E>,
}

impl<E: Env> PoolWorker for StagedWorker<E> {
    type Tensor = E::Tensor;
    type Request = ();
    type Output = Memory<E::Tensor>;

    fn set_policy(&mut self, actor: Box<dyn Actor<Tensor = E::Tensor>>) {
        self.worker.actor = Some(actor);
    }

    fn reset(&mut self) -> Result<()> {
        *self.state.lock().unwrap() = self.worker.env.reset(sample_u64())?;
        Ok(())
    }

    fn collect(&mut self, (): ()) -> Result<Memory<E::Tensor>> {
        let mut guard = self.state.lock().unwrap();
        self.worker.step(guard.as_mut())
    }
}

struct DirectWorkerPool<E: Env> {
    buffers: ArrayHandle<TrajectoryBuffer<E::Tensor>>,
    workers: Workers<LocalStateWorker<E>>,
}

impl<E: Env + 'static> DirectWorkerPool<E> {
    fn build(
        env_builder: &EnvBuilderType<E>,
        execution_mode: SamplerExecutionMode,
    ) -> Result<Self> {
        let (buffers, handles) =
            bimodal_array(vec![TrajectoryBuffer::default(); env_builder.num_envs()]);
        let factories = handles.into_iter().enumerate().map(|(idx, buffer)| {
            let env_builder = env_builder.clone();
            move || {
                let (worker, state) = Worker::build(&env_builder, idx)?;
                Ok(LocalStateWorker {
                    state,
                    worker,
                    buffer,
                })
            }
        });
        let workers = Workers::build(factories, execution_mode)?;
        Ok(Self { buffers, workers })
    }

    fn collect(&mut self, bound: RolloutMode) -> Result<()> {
        self.workers
            .collect_into(0..self.workers.len(), bound, |_, ()| {})
    }
}

struct StagedWorkerPool<E: Env> {
    states: ArrayHandle<E::Tensor>,
    workers: Workers<StagedWorker<E>>,
    buffers: Vec<TrajectoryBuffer<E::Tensor>>,
    obs_normalizer: Normalizer<E::Tensor>,
}

type StagedWorkers<E> = (ArrayHandle<<E as Env>::Tensor>, Workers<StagedWorker<E>>);

impl<E: Env + 'static> StagedWorkerPool<E> {
    fn build(
        env_builder: &EnvBuilderType<E>,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Normalizer<E::Tensor>,
    ) -> Result<Self> {
        let n_envs = env_builder.num_envs();
        let (states, workers) = match execution_mode {
            SamplerExecutionMode::SingleThreaded => Self::build_inline_workers(env_builder)?,
            SamplerExecutionMode::MultiThreaded => Self::build_thread_workers(env_builder)?,
        };
        let mut pool = Self {
            states,
            workers,
            buffers: vec![TrajectoryBuffer::default(); n_envs],
            obs_normalizer,
        };
        pool.normalize_states()?;
        Ok(pool)
    }

    fn build_inline_workers(env_builder: &EnvBuilderType<E>) -> Result<StagedWorkers<E>> {
        let mut workers = Vec::with_capacity(env_builder.num_envs());
        let mut states = Vec::with_capacity(env_builder.num_envs());
        for idx in 0..env_builder.num_envs() {
            let (worker, state) = Worker::build(env_builder, idx)?;
            workers.push(worker);
            states.push(state);
        }
        let (states, handles) = bimodal_array(states);
        let workers = workers
            .into_iter()
            .zip(handles)
            .map(|(worker, state)| StagedWorker { state, worker })
            .collect();
        Ok((states, Workers::Inline(workers)))
    }

    fn build_thread_workers(env_builder: &EnvBuilderType<E>) -> Result<StagedWorkers<E>> {
        let n_envs = env_builder.num_envs();
        let mut workers = Vec::with_capacity(n_envs);
        // On failure, drop these senders before joining workers that may still
        // be waiting for their state handles.
        let mut state_senders = Vec::with_capacity(n_envs);
        let mut initial_states = Vec::with_capacity(n_envs);
        for idx in 0..n_envs {
            let (initial_tx, initial_rx) = unbounded();
            let (state_tx, state_rx) = unbounded();
            let env_builder = env_builder.clone();
            let worker = ThreadWorker::spawn(move || {
                let (worker, initial_state) = Worker::build(&env_builder, idx)?;
                initial_tx.send(initial_state).map_err(Error::wrap)?;
                let state = state_rx.recv().map_err(Error::wrap)?;
                Ok(StagedWorker { state, worker })
            });
            workers.push(worker);
            state_senders.push(state_tx);
            let state = match initial_rx.recv() {
                Ok(state) => state,
                Err(error) => {
                    // Retrieve the original build/reset error from the worker.
                    workers.last().unwrap().wait_ready()?;
                    return Err(Error::wrap(error));
                }
            };
            initial_states.push(state);
        }
        let (states, handles) = bimodal_array(initial_states);
        for (sender, handle) in state_senders.into_iter().zip(handles) {
            sender.send(handle).map_err(Error::wrap)?;
        }
        for worker in &workers {
            worker.wait_ready()?;
        }
        Ok((states, Workers::Threaded(workers)))
    }

    fn normalize_states(&mut self) -> Result<()> {
        self.obs_normalizer
            .apply_slice_in_place(&mut self.states.lock().unwrap())?;
        Ok(())
    }

    fn step(&mut self) -> Result<()> {
        let indices = 0..self.workers.len();
        let mut memories = MultiMemory::with_capacity(indices.len());
        self.workers
            .collect_into(indices.clone(), (), |_, memory| {
                memories.push_memory(memory);
            })?;
        self.normalize_states()?;
        self.finish_step(indices, memories, |_, _| {})
    }

    fn step_indexed(&mut self, indices: &[usize], episodes: &mut [usize]) -> Result<()> {
        let mut memories = MultiMemory::with_capacity(indices.len());
        self.workers
            .collect_into(indices.iter().copied(), (), |_, memory| {
                memories.push_memory(memory);
            })?;
        {
            let mut states = self.states.lock().unwrap();
            let mut active_states = indices
                .iter()
                .map(|&idx| states[idx].clone())
                .collect::<Vec<_>>();
            self.obs_normalizer
                .apply_slice_in_place(&mut active_states)?;
            for (&idx, state) in indices.iter().zip(active_states) {
                states[idx] = state;
            }
        }
        self.finish_step(indices.iter().copied(), memories, |idx, done| {
            episodes[idx] += usize::from(done);
        })
    }

    fn finish_step(
        &mut self,
        indices: impl Iterator<Item = usize>,
        mut memories: MultiMemory<E::Tensor>,
        mut on_done: impl FnMut(usize, bool),
    ) -> Result<()> {
        self.obs_normalizer
            .with_mode(NormalizerMode::ReadOnly)
            .apply_slice_in_place(memories.next_states_mut())?;
        for (idx, memory) in indices.zip(memories.into_stored_memories_iter()) {
            on_done(idx, memory.is_done());
            self.buffers[idx].push(memory);
        }
        Ok(())
    }

    fn collect(&mut self, bound: RolloutMode) -> Result<()> {
        match bound {
            RolloutMode::StepBound { n_steps } => {
                for _ in 0..n_steps.get() {
                    self.step()?;
                }
            }
            RolloutMode::EpisodeBound { n_episodes } => {
                let mut episodes = vec![0; self.workers.len()];
                loop {
                    let indices = episodes
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &count)| (count < n_episodes.get()).then_some(idx))
                        .collect::<Vec<_>>();
                    if indices.is_empty() {
                        break;
                    }
                    self.step_indexed(&indices, &mut episodes)?;
                }
            }
        }
        Ok(())
    }
}

enum Pool<E: Env> {
    Direct(DirectWorkerPool<E>),
    Staged(StagedWorkerPool<E>),
}

enum TrajectoryViews<'a, T: R2lTensor> {
    Direct(ArrayMap<'a, TrajectoryView<'a, T>>),
    Staged(Vec<TrajectoryView<'a, T>>),
}

impl<'a, T: R2lTensor> AsRef<[TrajectoryView<'a, T>]> for TrajectoryViews<'a, T> {
    fn as_ref(&self) -> &[TrajectoryView<'a, T>] {
        match self {
            Self::Direct(views) => views.as_ref(),
            Self::Staged(views) => views.as_ref(),
        }
    }
}

enum BuffersMut<'a, T: R2lTensor> {
    Direct(ArrayGuard<'a, TrajectoryBuffer<T>>),
    Staged(&'a mut [TrajectoryBuffer<T>]),
}

impl<T: R2lTensor> AsMut<[TrajectoryBuffer<T>]> for BuffersMut<'_, T> {
    fn as_mut(&mut self) -> &mut [TrajectoryBuffer<T>] {
        match self {
            Self::Direct(buffers) => buffers.as_mut(),
            Self::Staged(buffers) => buffers,
        }
    }
}

/// Mutable sampler state exposed to collection hooks.
pub struct SamplerCore<E: Env + 'static> {
    pool: Pool<E>,
}

impl<E: Env + 'static> SamplerCore<E> {
    /// Collects a bounded rollout, appending to the current buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if an environment, normalization, or worker operation fails.
    pub fn collect(&mut self, bound: RolloutMode) -> Result<()> {
        match &mut self.pool {
            Pool::Direct(pool) => pool.collect(bound),
            Pool::Staged(pool) => pool.collect(bound),
        }
    }

    /// Installs a clone of the actor on each worker.
    ///
    /// # Errors
    ///
    /// Returns an error if a worker is interrupted.
    pub fn set_policy<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, actor: &A) -> Result<()> {
        match &mut self.pool {
            Pool::Direct(pool) => pool.workers.set_policy(actor),
            Pool::Staged(pool) => pool.workers.set_policy(actor),
        }
    }

    /// Borrows the output buffers, holding the array lock for direct collection.
    ///
    /// # Panics
    ///
    /// Panics if a direct worker still holds a buffer lock.
    pub fn buffers_mut(&mut self) -> impl AsMut<[TrajectoryBuffer<E::Tensor>]> {
        match &mut self.pool {
            Pool::Direct(pool) => BuffersMut::Direct(pool.buffers.lock().unwrap()),
            Pool::Staged(pool) => BuffersMut::Staged(&mut pool.buffers),
        }
    }

    /// Clears trajectories while retaining their allocated storage.
    ///
    /// # Panics
    ///
    /// Panics if a direct worker still holds a buffer lock.
    pub fn clear_buffers(&mut self) {
        self.buffers_mut()
            .as_mut()
            .iter_mut()
            .for_each(TrajectoryBuffer::clear);
    }

    /// Resets environments, normalizes initial observations, and clears trajectories.
    ///
    /// # Errors
    ///
    /// Returns an error if resetting, normalization, or worker communication fails.
    pub fn reset_all_envs(&mut self) -> Result<()> {
        match &mut self.pool {
            Pool::Direct(pool) => pool.workers.reset()?,
            Pool::Staged(pool) => {
                pool.workers.reset()?;
                pool.normalize_states()?;
            }
        }
        self.clear_buffers();
        Ok(())
    }

    /// Borrows collected trajectories in environment order.
    ///
    /// # Panics
    ///
    /// Panics if a direct worker still holds a buffer lock.
    pub fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, E::Tensor>]> {
        match &mut self.pool {
            Pool::Direct(direct) => {
                let trajectory_views = direct
                    .buffers
                    .lock_map(|buffer| buffer.to_trajectory_view())
                    .unwrap();
                TrajectoryViews::Direct(trajectory_views)
            }
            Pool::Staged(staged) => {
                let trajectory_views = staged
                    .buffers
                    .iter()
                    .map(|buffer| buffer.to_trajectory_view())
                    .collect::<Vec<_>>();
                TrajectoryViews::Staged(trajectory_views)
            }
        }
    }
}

/// Controls collection bounds and can inspect or transform the output buffers.
pub trait SamplerHook {
    /// Environment type sampled by the hook's sampler.
    type E: Env + 'static;

    /// Returns the next collection instruction.
    fn hook(&mut self, core: &mut SamplerCore<Self::E>) -> SamplerHookResult;

    /// Resets hook state before a new training or evaluation run.
    fn reset(&mut self) {}
}

/// Hook-controlled rollout collector with direct or observation-normalizing workers.
pub struct RolloutSampler<E: Env + 'static, H: SamplerHook<E = E>> {
    core: SamplerCore<E>,
    hook: H,
}

impl<E: Env + 'static, H: SamplerHook<E = E>> RolloutSampler<E, H> {
    fn new(pool: Pool<E>, hook: H) -> Self {
        Self {
            core: SamplerCore { pool },
            hook,
        }
    }

    /// Builds workers that collect directly into trajectory buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if an environment cannot be built or reset, or a worker is interrupted.
    pub fn build_direct(
        env_builder: &EnvBuilderType<E>,
        hook: H,
        execution_mode: SamplerExecutionMode,
    ) -> Result<Self> {
        Ok(Self::new(
            Pool::Direct(DirectWorkerPool::build(env_builder, execution_mode)?),
            hook,
        ))
    }

    /// Builds workers whose observations are normalized between steps.
    ///
    /// The supplied normalizer's update/read-only mode is preserved.
    ///
    /// # Errors
    ///
    /// Returns an error if environment initialization, normalization, or worker startup fails.
    pub fn build_staged(
        env_builder: &EnvBuilderType<E>,
        hook: H,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Normalizer<E::Tensor>,
    ) -> Result<Self> {
        Ok(Self::new(
            Pool::Staged(StagedWorkerPool::build(
                env_builder,
                execution_mode,
                obs_normalizer,
            )?),
            hook,
        ))
    }

    /// Returns a handle to shared observation statistics, or `None` for direct collection.
    pub fn obs_normalizer(&self) -> Option<Normalizer<E::Tensor>> {
        match &self.core.pool {
            Pool::Direct(_) => None,
            Pool::Staged(pool) => Some(pool.obs_normalizer.clone()),
        }
    }

    /// Builds a homogeneous direct sampler from a shared environment factory.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_envs` is zero or worker initialization fails.
    pub fn build_direct_from_env_builder(
        env_builder: Arc<dyn EnvBuilder<Env = E>>,
        num_envs: usize,
        hook: H,
        execution_mode: SamplerExecutionMode,
    ) -> Result<Self> {
        Self::build_direct(
            &EnvBuilderType::homogeneous_shared(env_builder, num_envs)?,
            hook,
            execution_mode,
        )
    }

    /// Builds a homogeneous staged sampler from a shared environment factory.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_envs` is zero, worker initialization fails, or observations
    /// cannot be normalized.
    pub fn build_staged_from_env_builder(
        env_builder: Arc<dyn EnvBuilder<Env = E>>,
        num_envs: usize,
        hook: H,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Normalizer<E::Tensor>,
    ) -> Result<Self> {
        Self::build_staged(
            &EnvBuilderType::homogeneous_shared(env_builder, num_envs)?,
            hook,
            execution_mode,
            obs_normalizer,
        )
    }
}

impl<E: Env + 'static, H: SamplerHook<E = E>> Sampler for RolloutSampler<E, H> {
    type Tensor = E::Tensor;

    fn reset_all_envs(&mut self) -> Result<()> {
        self.core.reset_all_envs()?;
        self.hook.reset();
        Ok(())
    }

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> Result<()> {
        self.core.clear_buffers();
        self.core.set_policy(&actor)?;
        while let SamplerHookResult::Bound(bound) = self.hook.hook(&mut self.core) {
            self.core.collect(bound)?;
        }
        Ok(())
    }

    fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, Self::Tensor>]> {
        self.core.trajectory_views()
    }
}
