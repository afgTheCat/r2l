use std::marker::PhantomData;

use bimodal_array::ArrayHandle;
use r2l_core::{
    buffers::buffer::TrajectoryBuffer,
    env::Env,
    tensor::{R2lTensor, RunningMeanTensor},
};
use r2l_sampler::{RolloutMode, SamplerHook, SamplerHookResult, worker::WorkerPool};

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
        const EPS: f32 = 1e-8;
        const CLIP_OBS: f32 = 10.0;

        if observations.is_empty() {
            return observations;
        }

        let mut batch_data = Vec::new();
        let mut feature_shape = None;
        for observation in &observations {
            let (data, shape) = observation.to_vec_and_shape();
            feature_shape = Some(shape);
            batch_data.extend(data);
        }

        let mut batch_shape = vec![observations.len()];
        batch_shape.extend(feature_shape.unwrap_or_default());
        let batch = E::Tensor::from_vec_and_shape(batch_data, batch_shape);
        self.rm.update(&batch).unwrap();

        let (mean, _) = self.rm.mean.to_vec_and_shape();
        let (var, _) = self.rm.var.to_vec_and_shape();

        observations
            .into_iter()
            .map(|observation| {
                let (data, shape) = observation.to_vec_and_shape();
                let normalized = data
                    .into_iter()
                    .zip(mean.iter().zip(var.iter()))
                    .map(|(value, (mean, var))| {
                        ((value - *mean) / (var + EPS).sqrt()).clamp(-CLIP_OBS, CLIP_OBS)
                    })
                    .collect();
                E::Tensor::from_vec_and_shape(normalized, shape)
            })
            .collect()
    }
}

impl<E: Env<Tensor: RunningMeanTensor>> SamplerHook for ObservationNormalizerHook<E> {
    type E = E;

    fn hook(
        &mut self,
        buffer: &mut ArrayHandle<TrajectoryBuffer<<Self::E as Env>::Tensor>>,
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
            let last_states = worker_pool.get_last_states().unwrap();
            let normalized_observations = self.normalize_observations(last_states);
            let mut buffers = buffer.lock().unwrap();
            for (buffer, next_state) in buffers.iter_mut().zip(normalized_observations.into_iter())
            {
                buffer.replace_last_next_state(next_state);
            }
            SamplerHookResult::Stop
        }
    }
}
