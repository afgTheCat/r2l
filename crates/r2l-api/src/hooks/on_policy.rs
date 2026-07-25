use std::marker::PhantomData;

use anyhow::Result;
use r2l_core::{
    HookResult,
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
        /// Number of rollout collections after which training stops.
        total_rollouts: usize,
        /// Number of rollout collections completed so far.
        current_rollout: usize,
    },
    /// Stop after at least `total_steps` sampled environment steps.
    TotalStepBound {
        /// Number of sampled steps after which training stops.
        total_steps: usize,
        /// Number of sampled steps completed so far.
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

/// Learning-rate policy applied over the progress of an on-policy training run.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule {
    /// Keep the learning rate fixed throughout training.
    Constant(f64),
    /// Decay the initial learning rate linearly to zero.
    Linear(f64),
}

impl LearningRateSchedule {
    /// Returns the learning rate for the remaining fraction of training.
    pub fn value(self, progress_remaining: f64) -> f64 {
        match self {
            Self::Constant(learning_rate) => learning_rate,
            Self::Linear(initial_learning_rate) => {
                initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
            }
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
    learning_rate_schedule: Option<LearningRateSchedule>,
    evaluator: Option<BestActorEvaluator<A::Actor, S2>>,
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
            learning_rate_schedule: None,
            evaluator,
            _phantom: PhantomData,
        }
    }

    /// Applies a learning-rate schedule over the configured training duration.
    pub fn with_learning_rate_schedule(
        mut self,
        learning_rate_schedule: LearningRateSchedule,
    ) -> Self {
        self.learning_rate_schedule = Some(learning_rate_schedule);
        self
    }

    fn mark_progress(&mut self, runtime: &mut OnPolicyRuntime<A, S, C>) {
        match &mut self.learning_schedule {
            LearningSchedule::RolloutBound {
                current_rollout, ..
            } => *current_rollout += 1,
            LearningSchedule::TotalStepBound { current_step, .. } => {
                let rollouts = runtime.trajectory_containers();
                let rollout_steps: usize = rollouts.as_ref().iter().map(|e| e.actions.len()).sum();
                *current_step += rollout_steps;
            }
        }
    }

    fn progress_remaining(&self) -> f64 {
        match self.learning_schedule {
            LearningSchedule::RolloutBound {
                total_rollouts,
                current_rollout,
            } => 1.0 - current_rollout as f64 / total_rollouts as f64,
            LearningSchedule::TotalStepBound {
                total_steps,
                current_step,
            } => 1.0 - current_step as f64 / total_steps as f64,
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
        self.mark_progress(runtime);
        if let Some(learning_rate_schedule) = self.learning_rate_schedule {
            let learning_rate = match learning_rate_schedule {
                LearningRateSchedule::Constant(learning_rate) => learning_rate,
                LearningRateSchedule::Linear(initial_learning_rate) => {
                    let progress_remaining = self.progress_remaining();
                    initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
                }
            };
            runtime.agent.set_learning_rate(learning_rate);
        }
        HookResult::Continue
    }

    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult {
        if let Some(evaluator) = &mut self.evaluator {
            evaluator.eval(runtime);
        }
        if self.progress_remaining() <= 0. {
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
