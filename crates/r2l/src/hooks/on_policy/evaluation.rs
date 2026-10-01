use r2l_core::{
    HookResult,
    env::Env,
    error::Error,
    models::{Actor, ToSafetensors},
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
    tensor::R2lTensor,
};

use crate::{evaluator::BestPolicyEvaluator, hooks::progress::SharedTrainingProgress};

pub(crate) enum ScheduledEvaluator<A: Actor, E: Env> {
    Disabled,
    Enabled {
        evaluator: BestPolicyEvaluator<A, E>,
        rollouts_per_evaluation: usize,
        progress: SharedTrainingProgress,
        avg_reward_threshold: Option<f32>,
    },
}

impl<A: Actor + Clone + ToSafetensors, E: Env<Tensor: R2lTensor>> ScheduledEvaluator<A, E> {
    pub(crate) fn disabled() -> Self {
        Self::Disabled
    }

    /// Creates an evaluator scheduled using shared training progress.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - Evaluates and tracks the best policy.
    /// * `rollouts_per_evaluation` - Number of training rollouts between evaluations.
    /// * `progress` - Training progress used to determine when evaluation is due.
    /// * `avg_reward_threshold` - Stop when the average episode reward reaches this value, or
    ///   `None` to evaluate without a reward-based stop condition.
    pub(crate) fn new(
        evaluator: BestPolicyEvaluator<A, E>,
        rollouts_per_evaluation: usize,
        progress: SharedTrainingProgress,
        avg_reward_threshold: Option<f32>,
    ) -> Self {
        assert!(
            rollouts_per_evaluation > 0,
            "rollouts per evaluation must be greater than zero"
        );
        Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
            progress,
            avg_reward_threshold,
        }
    }

    pub(super) fn evaluate<AG: Agent<Actor = A>, S: Sampler<Tensor = E::Tensor>>(
        &mut self,
        runtime: &mut OnPolicyRuntime<AG, S>,
    ) -> Result<HookResult, Error> {
        let Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
            progress,
            avg_reward_threshold,
        } = self
        else {
            return Ok(HookResult::Continue);
        };
        let completed_rollouts = progress.borrow().completed_rollouts();
        if !completed_rollouts.is_multiple_of(*rollouts_per_evaluation) {
            return Ok(HookResult::Continue);
        }
        let evaluation_result = evaluator.evaluate(runtime)?;
        if avg_reward_threshold.is_some_and(|t| t <= evaluation_result.avg_reward()) {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    pub(super) fn finish_training(&self) -> Result<(), Error> {
        let Self::Enabled { evaluator, .. } = self else {
            return Ok(());
        };
        evaluator.finish_training()
    }
}
