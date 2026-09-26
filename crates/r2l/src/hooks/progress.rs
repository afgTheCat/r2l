use std::{cell::RefCell, rc::Rc};

use r2l_core::{
    HookResult,
    buffers::TrajectoryBatch,
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
};
use r2l_sampler::RolloutMode;

/// Stop policy for the on-policy training loop.
#[derive(Debug, Clone, Copy)]
pub enum TrainingLimit {
    /// Stop after `total_rollouts` completed rollouts.
    RolloutBound {
        /// Number of rollouts after which training stops.
        total_rollouts: usize,
    },
    /// Stop after at least `total_steps` sampled environment steps.
    TotalStepBound {
        /// Number of sampled steps after which training stops.
        total_steps: usize,
    },
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
        assert!(total_steps > 0, "total steps must be greater than zero");
        Self::TotalStepBound { total_steps }
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
        assert!(
            total_rollouts > 0,
            "total rollouts must be greater than zero"
        );
        Self::RolloutBound { total_rollouts }
    }
}

#[derive(Default)]
struct TrainingCounters {
    completed_rollouts: usize,
    steps_taken: usize,
}

pub(crate) struct TrainingProgress {
    counters: TrainingCounters,
    training_limit: TrainingLimit,
    rollout_mode: RolloutMode,
    n_envs: usize,
}

pub(crate) type SharedTrainingProgress = Rc<RefCell<TrainingProgress>>;

impl TrainingProgress {
    /// Creates progress shared by the training, agent, and sampler hooks.
    ///
    /// # Arguments
    ///
    /// * `training_limit` - Determines when the training loop stops.
    /// * `rollout_mode` - Collection bound applied to each environment.
    /// * `n_envs` - Number of training environments, used to estimate rollout count.
    pub(crate) fn shared(
        training_limit: TrainingLimit,
        rollout_mode: RolloutMode,
        n_envs: usize,
    ) -> SharedTrainingProgress {
        assert!(
            n_envs > 0,
            "number of environments must be greater than zero"
        );
        Rc::new(RefCell::new(Self {
            counters: TrainingCounters::default(),
            training_limit,
            rollout_mode,
            n_envs,
        }))
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

    pub(super) fn rollout_mode(&self) -> RolloutMode {
        self.rollout_mode
    }

    pub(crate) fn total_rollouts(&self) -> Option<usize> {
        match self.training_limit {
            TrainingLimit::RolloutBound { total_rollouts } => Some(total_rollouts),
            TrainingLimit::TotalStepBound { total_steps } => match self.rollout_mode {
                RolloutMode::StepBound { n_steps } => n_steps
                    .checked_mul(self.n_envs)
                    .filter(|steps| *steps > 0)
                    .map(|steps| total_steps.div_ceil(steps)),
                RolloutMode::EpisodeBound { .. } => None,
            },
        }
    }

    /// Remaining training fraction, including the latest collection even before learning.
    pub(crate) fn progress_remaining(&self) -> f64 {
        let remaining = match self.training_limit {
            TrainingLimit::RolloutBound { total_rollouts } => {
                1.0 - self.counters.completed_rollouts as f64 / total_rollouts as f64
            }
            TrainingLimit::TotalStepBound { total_steps } => {
                1.0 - self.counters.steps_taken as f64 / total_steps as f64
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
}
