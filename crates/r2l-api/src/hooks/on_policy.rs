use std::marker::PhantomData;

use anyhow::Result;
use r2l_core::{
    HookResult,
    buffers::TrajectoryContainer,
    on_policy::algorithm::{Agent, OnPolicyAlgorithmHooks, Sampler},
};

/// Training-stop policy for [`DefaultOnPolicyAlgorithmHooks`].
///
/// This determines when the outer on-policy training loop should terminate,
/// either after a fixed number of rollouts or after a fixed number of sampled
/// environment steps.
#[derive(Debug, Clone, Copy)]
pub enum LearningSchedule {
    /// Stop after `total_rollouts` completed rollout collections.
    RolloutBound {
        total_rollouts: usize,
        current_rollout: usize,
    },
    /// Stop after at least `total_steps` sampled environment steps.
    TotalStepBound {
        total_steps: usize,
        current_step: usize,
    },
}

impl LearningSchedule {
    /// Creates a schedule bounded by total sampled environment steps.
    pub fn total_step_bound(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps,
            current_step: 0,
        }
    }

    /// Creates a schedule bounded by completed rollout collections.
    pub fn rollout_bound(total_rollouts: usize) -> Self {
        Self::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }
    }
}

/// Default outer-loop hooks used by high-level on-policy algorithm builders.
///
/// This hook is responsible for lifecycle behavior around training rather than
/// algorithm-specific loss logic. It tracks rollout progress, applies the
/// configured [`LearningSchedule`] to decide when training should stop, and
/// shuts down both the agent and sampler when the algorithm exits.
///
/// [`A2CAlgorithmBuilder`](crate::A2CAlgorithmBuilder) and
/// [`PPOAlgorithmBuilder`](crate::PPOAlgorithmBuilder) install this hook by
/// default when building an [`OnPolicyAlgorithm`](crate::OnPolicyAlgorithm).
pub struct DefaultOnPolicyAlgorithmHooks<A: Agent, S: Sampler> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _phantom: PhantomData<(A, S)>,
}

impl<A: Agent, S: Sampler> DefaultOnPolicyAlgorithmHooks<A, S> {
    /// Creates the default outer-loop hooks for the given learning schedule.
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _phantom: PhantomData,
        }
    }
}

impl<A: Agent, S: Sampler> OnPolicyAlgorithmHooks for DefaultOnPolicyAlgorithmHooks<A, S> {
    type A = A;
    type S = S;

    fn init_hook(&mut self) -> HookResult {
        HookResult::Continue
    }

    fn post_rollout_hook(
        &mut self,
        rollouts: &[<Self::S as Sampler>::TrajectoryContainer],
    ) -> HookResult {
        self.rollout_idx += 1;
        let should_stop = match &mut self.learning_schedule {
            LearningSchedule::RolloutBound {
                total_rollouts,
                current_rollout,
            } => {
                *current_rollout += 1;
                current_rollout >= total_rollouts
            }
            LearningSchedule::TotalStepBound {
                total_steps,
                current_step,
            } => {
                let rollout_steps: usize = rollouts.iter().map(|e| e.actions().count()).sum();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        };
        if should_stop {
            HookResult::Break
        } else {
            HookResult::Continue
        }
    }

    fn post_training_hook(&mut self, _policy: <Self::A as Agent>::Actor) -> HookResult {
        HookResult::Continue
    }

    fn shutdown_hook(&mut self, agent: &mut Self::A, sampler: &mut Self::S) -> Result<()> {
        agent.shutdown();
        sampler.shutdown();
        Ok(())
    }
}
