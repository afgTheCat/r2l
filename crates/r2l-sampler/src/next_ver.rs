use bimodal_array::{ArrayHandle, ArrayMap, ElementHandle};
use r2l_core::buffers::buffer::TrajectoryView;
use r2l_core::buffers::{Memory, MultiMemory};
use r2l_core::env::Snapshot;
use r2l_core::env::normalizer::{Normalizer, NormalizerMode};
use r2l_core::error::{Error, Result};
use r2l_core::on_policy::algorithm::Sampler;
use r2l_core::rng::sample_u64;
use r2l_core::{buffers::buffer::TrajectoryBuffer, env::Env, models::Actor, tensor::R2lTensor};

use crate::RolloutMode;

// there are two ways of going about this one:
// workers write to the buffer or they have something that normalizes shit

struct Worker<E: Env> {
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<E: Env> Worker<E> {
    fn get_policy(&mut self) -> Result<&mut Box<dyn Actor<Tensor = E::Tensor>>> {
        self.actor.as_mut().ok_or_else(|| {
            Error::invalid_state("step staged sampler worker", "no policy has been installed")
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

// State is locally stored, buffer is filled instantly. good for paralell work
struct LocalStateWorker<E: Env> {
    state: E::Tensor,
    worker: Worker<E>,
    buffer: ElementHandle<TrajectoryBuffer<E::Tensor>>,
}

impl<E: Env> LocalStateWorker<E> {
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

// States guared with an element guard
struct StagedWorker<E: Env> {
    state: ElementHandle<E::Tensor>,
    worker: Worker<E>,
}

impl<E: Env> StagedWorker<E> {
    fn step(&mut self) -> Result<Memory<E::Tensor>> {
        let mut guard = self.state.lock().unwrap();
        self.worker.step(guard.as_mut())
    }
}

struct DirectWorkerPool<E: Env> {
    buffers: ArrayHandle<TrajectoryBuffer<E::Tensor>>,
}

// option a: it has the array handle. It does not even have to be
// option b
struct StagedWorkerPool<E: Env> {
    states: ArrayHandle<E::Tensor>,
    workers: Vec<StagedWorker<E>>, // should be vec or thread
    buffers: Vec<TrajectoryBuffer<E::Tensor>>,
    obs_normalizer: Normalizer<E::Tensor>,
}

impl<E: Env> StagedWorkerPool<E> {
    // only do Step  bound for now
    fn collect(&mut self, bound: RolloutMode) -> Result<()> {
        match bound {
            RolloutMode::StepBound { n_steps } => {
                let n_workers = self.workers.len();
                for _ in 0..n_steps.get() {
                    let mut multi_memory = MultiMemory::with_capacity(n_workers);
                    for worker in &mut self.workers {
                        let memory = worker.step()?;
                        multi_memory.push_memory(memory);
                    }
                    let mut last_states = self.states.lock().unwrap();
                    self.obs_normalizer.apply_slice_in_place(&mut last_states)?;
                    self.obs_normalizer
                        .with_mode(NormalizerMode::ReadOnly)
                        .apply_slice_in_place(multi_memory.next_states_mut())?;
                    let memories = multi_memory.into_stored_memories();
                    for (idx, memory) in memories.into_iter().enumerate() {
                        self.buffers[idx].push(memory);
                    }
                }
            }
            RolloutMode::EpisodeBound { n_episodes } => {
                // something similiar only with the step_indexed variant
                todo!()
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

impl<E: Env> Pool<E> {
    fn collect(&mut self) {}

    fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, E::Tensor>]> {
        match self {
            Self::Direct(direct) => {
                let trajectory_views = direct
                    .buffers
                    .lock_map(|buffer| buffer.to_trajectory_view())
                    .unwrap();
                TrajectoryViews::Direct(trajectory_views)
            }
            Self::Staged(staged) => {
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

struct ReSampler<E: Env> {
    pool: Pool<E>,
}

impl<E: Env> Sampler for ReSampler<E> {
    type Tensor = E::Tensor;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> Result<()> {
        todo!()
    }

    fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, Self::Tensor>]> {
        self.pool.trajectory_views()
    }
}
