use std::{cell::RefCell, num::NonZeroUsize, rc::Rc};

use r2l_core::{
    HookResult,
    buffers::TrajectoryBatch,
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
};
use r2l_sampler::RolloutMode;
use serde::{Deserialize, Serialize};

/// Stop policy for the on-policy training loop.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrainingLimit {
    /// Stop after `total_rollouts` completed rollouts.
    RolloutBound {
        /// Number of rollouts after which training stops.
        total_rollouts: NonZeroUsize,
    },
    /// Stop after at least `total_steps` sampled environment steps.
    TotalStepBound {
        /// Number of sampled steps after which training stops.
        total_steps: NonZeroUsize,
    },
}

impl Default for TrainingLimit {
    fn default() -> Self {
        TrainingLimit::rollouts(50)
    }
}

impl TrainingLimit {
    /// Creates a schedule bounded by total sampled environment steps.
    ///
    /// # Arguments
    ///
    /// * `total_steps` - Minimum number of sampled steps after which training stops.
    ///
    /// # Panics
    ///
    /// Panics if `total_steps` is zero.
    #[must_use]
    pub fn steps(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps: NonZeroUsize::new(total_steps)
                .expect("total steps must be greater than zero"),
        }
    }

    /// Creates a schedule bounded by completed rollouts.
    ///
    /// # Arguments
    ///
    /// * `total_rollouts` - Number of completed rollouts after which training stops.
    ///
    /// # Panics
    ///
    /// Panics if `total_rollouts` is zero.
    #[must_use]
    pub fn rollouts(total_rollouts: usize) -> Self {
        Self::RolloutBound {
            total_rollouts: NonZeroUsize::new(total_rollouts)
                .expect("total rollouts must be greater than zero"),
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct TrainingCounters {
    completed_rollouts: usize,
    steps_taken: usize,
    n_evaluations: usize,
}

#[derive(Clone)]
pub(crate) struct TrainingProgress {
    counters: TrainingCounters,
    training_limit: TrainingLimit,
    total_rollouts: Option<usize>,
    n_envs: NonZeroUsize,
}

pub(crate) type SharedTrainingProgress = Rc<RefCell<TrainingProgress>>;

impl TrainingProgress {
    pub(crate) fn n_envs(&self) -> NonZeroUsize {
        self.n_envs
    }

    pub(crate) fn counters(&self) -> TrainingCounters {
        self.counters.clone()
    }

    pub(crate) fn restore_counters(&mut self, counters: TrainingCounters) {
        self.counters = counters;
    }

    /// Creates progress shared by the training and learning hooks.
    ///
    /// # Arguments
    ///
    /// * `training_limit` - Determines when the training loop stops.
    /// * `rollout_mode` - Collection bound used to calculate the total rollout count once.
    /// * `n_envs` - Number of training environments, also used to size rollout reward tracking.
    pub(crate) fn shared(
        training_limit: TrainingLimit,
        rollout_mode: RolloutMode,
        n_envs: NonZeroUsize,
    ) -> SharedTrainingProgress {
        let total_rollouts = match training_limit {
            TrainingLimit::RolloutBound { total_rollouts } => Some(total_rollouts.get()),
            TrainingLimit::TotalStepBound { total_steps } => match rollout_mode {
                RolloutMode::StepBound { n_steps } => n_steps
                    .checked_mul(n_envs)
                    .map(|steps| total_steps.get().div_ceil(steps.get())),
                RolloutMode::EpisodeBound { .. } => None,
            },
        };
        Self {
            counters: TrainingCounters::default(),
            training_limit,
            total_rollouts,
            n_envs,
        }
        .into_shared()
    }

    /// Wraps owned progress in fresh storage shared by the restored hooks.
    pub(crate) fn into_shared(self) -> SharedTrainingProgress {
        Rc::new(RefCell::new(self))
    }

    /// Records a collected rollout and its sampled steps before the learning pass.
    ///
    /// # Arguments
    ///
    /// * `runtime` - Supplies the trajectories collected across all environments.
    pub(crate) fn update_rollout_progress<A: Agent, S: Sampler>(
        &mut self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) {
        let rollouts = runtime.trajectory_containers();
        let steps_taken: usize = rollouts.as_ref().iter().map(TrajectoryBatch::len).sum();
        self.counters.steps_taken += steps_taken;
        self.counters.completed_rollouts += 1;
    }

    /// Number of completed collections, including any whose learning pass stopped or failed.
    pub(crate) fn completed_rollouts(&self) -> usize {
        self.counters.completed_rollouts
    }

    pub(crate) fn total_rollouts(&self) -> Option<usize> {
        self.total_rollouts
    }

    /// Remaining training fraction, including the latest collection even before learning.
    pub(crate) fn progress_remaining(&self) -> f64 {
        let remaining = match self.training_limit {
            TrainingLimit::RolloutBound { total_rollouts } => {
                1.0 - self.counters.completed_rollouts as f64 / total_rollouts.get() as f64
            }
            TrainingLimit::TotalStepBound { total_steps } => {
                1.0 - self.counters.steps_taken as f64 / total_steps.get() as f64
            }
        };
        remaining.clamp(0.0, 1.0)
    }

    pub(crate) fn progress_result(&self) -> HookResult {
        if self.progress_remaining() <= 0.0 {
            HookResult::Break
        } else {
            HookResult::Continue
        }
    }

    pub(crate) fn n_evaluations(&self) -> usize {
        self.counters.n_evaluations
    }

    pub(crate) fn increase_evaluations(&mut self) {
        self.counters.n_evaluations += 1;
    }
}
