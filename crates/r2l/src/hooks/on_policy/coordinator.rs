use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use r2l_core::{
    buffers::TrajectoryBatch,
    env::Env,
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
};
use r2l_sampler::{
    DirectSamplerCore, DirectSamplerHook, RolloutMode, SamplerHookResult, StagedSamplerCore,
    StagedSamplerHook,
};

use super::TrainingLimit;
use crate::utils::RewardNormalizer;

#[derive(Default)]
struct TrainingState {
    completed_rollouts: usize,
    steps_taken: usize,
}

pub(crate) type SharedCoordinator = Rc<RefCell<Coordinator>>;

pub(crate) struct Coordinator {
    training_state: TrainingState,
    training_limit: TrainingLimit,
    rollout_mode: RolloutMode,
    n_envs: usize,
}

impl Coordinator {
    /// Creates progress shared by the training, agent, and sampler hooks.
    ///
    /// # Arguments
    ///
    /// * `training_limit` - Determines when the training loop stops.
    /// * `rollout_mode` - Collection bound applied to each environment.
    /// * `n_envs` - Number of training environments, used to estimate rollout count.
    pub(crate) fn new(
        training_limit: TrainingLimit,
        rollout_mode: RolloutMode,
        n_envs: usize,
    ) -> Self {
        assert!(
            n_envs > 0,
            "number of environments must be greater than zero"
        );
        Self {
            training_state: TrainingState::default(),
            training_limit,
            rollout_mode,
            n_envs,
        }
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
        self.training_state.steps_taken += steps_taken;
        self.training_state.completed_rollouts += 1;
    }

    /// Number of completed collections, including any whose learning pass stopped or failed.
    pub(crate) fn completed_rollouts(&self) -> usize {
        self.training_state.completed_rollouts
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
                1.0 - self.training_state.completed_rollouts as f64 / total_rollouts as f64
            }
            TrainingLimit::TotalStepBound { total_steps } => {
                1.0 - self.training_state.steps_taken as f64 / total_steps as f64
            }
        };
        remaining.clamp(0.0, 1.0)
    }
}

enum Phase {
    Collect,
    Stop,
}

/// Marker for a sampler configured to collect a fixed number of steps.
pub struct Steps;
/// Marker for a sampler configured to collect a fixed number of episodes.
pub struct Episodes;

/// Step-bound specialization of the shared sampler hook.
pub type StepBoundHook<E> = SamplerHook<E, Steps>;
/// Episode-bound specialization of the shared sampler hook.
pub type EpisodeBoundHook<E> = SamplerHook<E, Episodes>;

/// Controls collection and optional reward normalization using shared progress.
pub struct SamplerHook<E: Env, Bound> {
    coordinator: SharedCoordinator,
    reward_normalizer: Option<RewardNormalizer>,
    phase: Phase,
    _p: PhantomData<(E, Bound)>,
}

impl<E: Env, Bound> SamplerHook<E, Bound> {
    /// Creates a sampler hook using the shared collection configuration.
    ///
    /// # Arguments
    ///
    /// * `coordinator` - Progress and collection settings shared with the training hooks.
    /// * `reward_normalizer` - Optional normalization for step-bound rollouts.
    pub(crate) fn new(
        coordinator: SharedCoordinator,
        reward_normalizer: Option<RewardNormalizer>,
    ) -> Self {
        Self {
            coordinator,
            reward_normalizer,
            phase: Phase::Collect,
            _p: PhantomData,
        }
    }

    fn reset_state(&mut self) {
        self.phase = Phase::Collect;
        if let Some(normalizer) = &mut self.reward_normalizer {
            normalizer.reset_returns();
        }
    }

    fn next_result(&mut self, normalize: impl FnOnce(&mut RewardNormalizer)) -> SamplerHookResult {
        let mode = self.coordinator.borrow().rollout_mode;
        match self.phase {
            Phase::Collect => {
                self.phase = Phase::Stop;
                SamplerHookResult::Bound(mode)
            }
            Phase::Stop => {
                self.phase = Phase::Collect;
                if mode.is_step_bound()
                    && let Some(normalizer) = &mut self.reward_normalizer
                {
                    normalize(normalizer);
                }
                SamplerHookResult::Stop
            }
        }
    }
}

impl<E: Env, Bound> DirectSamplerHook for SamplerHook<E, Bound> {
    type E = E;

    fn reset(&mut self) {
        self.reset_state();
    }

    fn hook(&mut self, core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        self.next_result(|normalizer| {
            let mut buffers = core.buffers_mut().lock().unwrap();
            normalizer.normalize(&mut buffers);
        })
    }
}

impl<E: Env, Bound> StagedSamplerHook for SamplerHook<E, Bound> {
    type E = E;

    fn reset(&mut self) {
        self.reset_state();
    }

    fn hook(&mut self, core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult {
        self.next_result(|normalizer| normalizer.normalize(core.buffers_mut()))
    }
}
