use r2l_core::{
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
    pub(crate) fn new(
        evaluator: BestPolicyEvaluator<A, E>,
        rollouts_per_evaluation: usize,
        progress: SharedTrainingProgress,
    ) -> Self {
        assert!(
            rollouts_per_evaluation > 0,
            "rollouts per evaluation must be greater than zero"
        );
        Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
            progress,
        }
    }

    pub(super) fn evaluate<AG: Agent<Actor = A>, S: Sampler<Tensor = E::Tensor>>(
        &mut self,
        runtime: &mut OnPolicyRuntime<AG, S>,
    ) -> Result<(), Error> {
        let Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
            progress,
        } = self
        else {
            return Ok(());
        };
        let completed_rollouts = progress.borrow().completed_rollouts();
        if completed_rollouts.is_multiple_of(*rollouts_per_evaluation) {
            return evaluator.evaluate(runtime);
        }
        Ok(())
    }

    pub(super) fn finish_training(&self) -> Result<(), Error> {
        let Self::Enabled { evaluator, .. } = self else {
            return Ok(());
        };
        evaluator.finish_training()
    }
}
