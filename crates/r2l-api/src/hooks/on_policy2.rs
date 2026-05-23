use std::marker::PhantomData;

use anyhow::Result;
use r2l_core::{
    HookResult,
    env::{Env, EnvBuilder},
    on_policy::algorithm2::{
        Agent2, OnPolicyAdapters2, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler2,
    },
};

use crate::{BestActorEvaluator, BestActorEvaluatorBuilder};

#[derive(Debug, Clone, Copy)]
pub enum LearningSchedule2 {
    RolloutBound {
        total_rollouts: usize,
        current_rollout: usize,
    },
    TotalStepBound {
        total_steps: usize,
        current_step: usize,
    },
}

impl LearningSchedule2 {
    pub fn total_step_bound(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps,
            current_step: 0,
        }
    }

    pub fn rollout_bound(total_rollouts: usize) -> Self {
        Self::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }
    }
}

pub struct DefaultOnPolicyAlgorithmHooks2<
    A: Agent2,
    S: Sampler2,
    C: OnPolicyAdapters2<A::Actor, S>,
    E: Env<Tensor = S::Tensor>,
> {
    learning_schedule: LearningSchedule2,
    evaluator: Option<BestActorEvaluator<E, A::Actor>>,
    should_stop: bool,
    _phantom: PhantomData<(A, S, C)>,
}

impl<A: Agent2, S: Sampler2, C: OnPolicyAdapters2<A::Actor, S>, E: Env<Tensor = S::Tensor>>
    DefaultOnPolicyAlgorithmHooks2<A, S, C, E>
where
    A::Tensor: From<S::Tensor>,
{
    pub fn new<EB: EnvBuilder<Env = E>>(
        learning_schedule: LearningSchedule2,
        evaluator_builder: Option<BestActorEvaluatorBuilder<EB>>,
    ) -> Self {
        Self {
            learning_schedule,
            evaluator: evaluator_builder.map(BestActorEvaluatorBuilder::build),
            should_stop: false,
            _phantom: PhantomData,
        }
    }
}

impl<A: Agent2, S: Sampler2, C: OnPolicyAdapters2<A::Actor, S>, E: Env<Tensor = S::Tensor>>
    OnPolicyAlgorithmHooks for DefaultOnPolicyAlgorithmHooks2<A, S, C, E>
where
    A::Tensor: From<S::Tensor>,
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
            LearningSchedule2::RolloutBound {
                total_rollouts,
                current_rollout,
            } => {
                *current_rollout += 1;
                current_rollout >= total_rollouts
            }
            LearningSchedule2::TotalStepBound {
                total_steps,
                current_step,
            } => {
                let rollouts = runtime.trajectory_containers::<A::Tensor>();
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
            let actor = runtime.actor();
            let adapted_actor = runtime.adapted_actor();
            evaluator.eval(adapted_actor, actor);
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
