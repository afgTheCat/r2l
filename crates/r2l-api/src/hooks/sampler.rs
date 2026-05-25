use std::marker::PhantomData;

use bimodal_array::ArrayHandle;
use candle_core::Tensor;
use r2l_core::{buffers::buffer::TrajectoryBuffer, env::Env, tensor::RunningMeanTensor};
use r2l_sampler::{
    RolloutMode, SamplerHook, SamplerHookResult,
    worker::{self, WorkerPool},
};

use crate::utils::running_mean2::RunningMeanStd2;

/// Sampler hook that requests rollout collection until a fixed number of
/// episodes has been scheduled.
///
/// The hook returns an episode-bound rollout mode once, then returns
/// [`SamplerHookResult::Stop`] on the next call so the outer sampler loop can
/// hand the collected data off for training.
pub struct EpisodeBoundHook<E: Env> {
    num_episodes: usize,
    episodes_scheduled: usize,
    _p: PhantomData<E>,
}

impl<E: Env> EpisodeBoundHook<E> {
    /// Creates an episode-bound sampler hook.
    pub fn new(num_episodes: usize) -> Self {
        Self {
            num_episodes,
            episodes_scheduled: 0,
            _p: PhantomData,
        }
    }
}

impl<E: Env> SamplerHook for EpisodeBoundHook<E> {
    type E = E;

    fn hook(
        &mut self,
        _buffer: &mut ArrayHandle<TrajectoryBuffer<<Self::E as Env>::Tensor>>,
        _worker_pool: &mut WorkerPool<Self::E>,
    ) -> SamplerHookResult {
        if self.episodes_scheduled == self.num_episodes {
            self.episodes_scheduled = 0;
            SamplerHookResult::Stop
        } else {
            self.episodes_scheduled = self.num_episodes;
            SamplerHookResult::Bound(RolloutMode::EpisodeBound {
                n_episodes: self.num_episodes,
            })
        }
    }
}

/// Sampler hook that requests rollout collection until a fixed number of steps
/// has been scheduled.
///
/// The hook returns a step-bound rollout mode once, then returns
/// [`SamplerHookResult::Stop`] on the next call so the outer sampler loop can
/// hand the collected data off for training.
pub struct StepBoundHook<E: Env> {
    num_steps: usize,
    steps_scheduled: usize,
    _p: PhantomData<E>,
}

impl<E: Env> StepBoundHook<E> {
    /// Creates a step-bound sampler hook.
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps,
            steps_scheduled: 0,
            _p: PhantomData,
        }
    }
}

impl<E: Env> SamplerHook for StepBoundHook<E> {
    type E = E;

    fn hook(
        &mut self,
        _buffer: &mut ArrayHandle<TrajectoryBuffer<<Self::E as Env>::Tensor>>,
        _worker_pool: &mut WorkerPool<Self::E>,
    ) -> SamplerHookResult {
        if self.steps_scheduled == self.num_steps {
            self.steps_scheduled = 0;
            SamplerHookResult::Stop
        } else {
            self.steps_scheduled = self.num_steps;
            SamplerHookResult::Bound(RolloutMode::StepBound {
                n_steps: self.num_steps,
            })
        }
    }
}

pub struct ObservationNormalizerHook<E: Env<Tensor: RunningMeanTensor>> {
    num_steps: usize,
    steps_scheduled: usize,
    rm: RunningMeanStd2<E::Tensor>,
}

impl<E: Env<Tensor: RunningMeanTensor>> ObservationNormalizerHook<E> {
    fn normalize_observations(&mut self, observations: Vec<E::Tensor>) -> Vec<E::Tensor> {
        todo!()
    }
}

impl<E: Env<Tensor: RunningMeanTensor>> SamplerHook for ObservationNormalizerHook<E> {
    type E = E;

    fn hook(
        &mut self,
        _buffer: &mut ArrayHandle<TrajectoryBuffer<<Self::E as Env>::Tensor>>,
        worker_pool: &mut WorkerPool<Self::E>,
    ) -> SamplerHookResult {
        if self.steps_scheduled < self.num_steps {
            let last_states = worker_pool
                .get_last_states()
                .or_else(|| Some(worker_pool.reset_envs_uninserted()))
                .unwrap();
            let normalized_observations = self.normalize_observations(last_states);
            worker_pool.set_last_states(normalized_observations);
            self.steps_scheduled += 1;
            SamplerHookResult::Bound(RolloutMode::StepBound { n_steps: 1 })
        } else {
            self.steps_scheduled = 0;
            SamplerHookResult::Stop
        }
    }
}
