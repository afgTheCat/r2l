use std::marker::PhantomData;

use anyhow::Result;
use r2l_core::{
    HookResult,
    buffers::TrajectoryBatch,
    env::Env,
    on_policy::algorithm::{
        Agent, OnPolicyAdapters, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler,
    },
    tensor::R2lTensor,
};

use crate::BestActorEvaluator;

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
/// configured [`LearningSchedule`] to decide when training should stop,
/// optionally evaluates the current actor, and shuts down the runtime when the
/// algorithm exits.
pub struct DefaultOnPolicyAlgorithmHooks<
    A: Agent,
    S: Sampler,
    C: OnPolicyAdapters<A::Actor, S>,
    E: Env<Tensor = S::Tensor>,
    S2: Sampler<Tensor = S::Tensor>,
> {
    learning_schedule: LearningSchedule,
    evaluator: Option<BestActorEvaluator<A::Actor, S2>>,
    should_stop: bool,
    _phantom: PhantomData<(A, S, C, E)>,
}

impl<
    A: Agent,
    S: Sampler<Tensor: R2lTensor>,
    C: OnPolicyAdapters<A::Actor, S>,
    E: Env<Tensor = S::Tensor>,
    S2: Sampler<Tensor = S::Tensor>,
> DefaultOnPolicyAlgorithmHooks<A, S, C, E, S2>
{
    /// Creates the default outer-loop hooks for the given learning schedule.
    pub fn new(
        learning_schedule: LearningSchedule,
        evaluator: Option<BestActorEvaluator<A::Actor, S2>>,
    ) -> Self {
        Self {
            learning_schedule,
            evaluator,
            should_stop: false,
            _phantom: PhantomData,
        }
    }
}

impl<
    A: Agent,
    S: Sampler<Tensor: R2lTensor>,
    C: OnPolicyAdapters<A::Actor, S>,
    E: Env<Tensor = S::Tensor>,
    S2: Sampler<Tensor = S::Tensor>,
> OnPolicyAlgorithmHooks for DefaultOnPolicyAlgorithmHooks<A, S, C, E, S2>
{
    type A = A;
    type S = S;
    type C = C;

    fn init_hook(
        &mut self,
        _runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult {
        HookResult::Continue
    }

    fn post_rollout_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult {
        self.should_stop = match &mut self.learning_schedule {
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
                let rollouts = runtime.trajectory_containers();
                let rollout_steps: usize =
                    rollouts.as_ref().iter().map(|e| e.actions().len()).sum();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        };

        HookResult::Continue
    }

    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult {
        if let Some(evaluator) = &mut self.evaluator {
            evaluator.eval(runtime);
        }
        if self.should_stop {
            HookResult::Break
        } else {
            HookResult::Continue
        }
    }

    fn shutdown_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> Result<()> {
        if let Some(evaluator) = &mut self.evaluator {
            evaluator.try_write_to_file()?;
            evaluator.shutdown();
        }
        runtime.shutdown();
        Ok(())
    }
}
