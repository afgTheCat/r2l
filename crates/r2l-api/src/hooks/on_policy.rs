use std::{marker::PhantomData, sync::Arc};

use anyhow::Result;
use r2l_core::{
    HookResult,
    buffers::TrajectoryContainer,
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::{
        Agent, OnPolicyAdapters, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler,
    },
};
use r2l_sampler::{EpisodeTrajectoryBound, R2lSampler};

struct Evaluator<E: Env, A: Actor> {
    sampler: R2lSampler<E, EpisodeTrajectoryBound<E::Tensor>>,
    best_actor: Option<A>,
    best_rewards: f32,
}

impl<E: Env, A: Actor> Evaluator<E, A> {
    fn new(builder: impl EnvBuilder<Env = E>) -> Self {
        let env_builder = EnvBuilderType::Homogenous {
            builder: Arc::new(builder),
            n_envs: 10,
        };
        Self {
            sampler: R2lSampler::build(
                env_builder,
                EpisodeTrajectoryBound::new(5),
                r2l_sampler::SamplerExecutionMode::Thread,
            ),
            best_rewards: f32::MIN,
            best_actor: None,
        }
    }

    // fn eval<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, actor: A) {
    //     self.sampler.reset_all_envs();
    //     let trajectories = self.sampler.collect_rollouts(actor);
    //     let avg_reward = trajectories
    //         .as_ref()
    //         .iter()
    //         .map(|x| x.rewards().sum::<f32>())
    //         .sum::<f32>();
    //     if avg_reward > self.best_rewards {
    //         self.best_rewards = avg_reward;
    //     }
    // }
}

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
pub struct DefaultOnPolicyAlgorithmHooks<
    A: Agent,
    S: Sampler,
    C: OnPolicyAdapters<A, S>,
    E: Env<Tensor = S::Tensor>,
> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    evaluator: Option<Evaluator<E, A::Actor>>,
    _phantom: PhantomData<(A, S, C)>,
}

impl<A: Agent, S: Sampler, C: OnPolicyAdapters<A, S>, E: Env<Tensor = S::Tensor>>
    DefaultOnPolicyAlgorithmHooks<A, S, C, E>
{
    /// Creates the default outer-loop hooks for the given learning schedule.
    pub fn new(
        learning_schedule: LearningSchedule,
        builder: Option<impl EnvBuilder<Env = E>>,
    ) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            evaluator: builder.map(|b| Evaluator::new(b)),
            _phantom: PhantomData,
        }
    }
}

impl<A: Agent, S: Sampler, C: OnPolicyAdapters<A, S>, E: Env<Tensor = S::Tensor>>
    OnPolicyAlgorithmHooks for DefaultOnPolicyAlgorithmHooks<A, S, C, E>
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
                let rollouts = runtime.trajectory_containers();
                let rollout_steps: usize =
                    rollouts.as_ref().iter().map(|e| e.actions().count()).sum();
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

    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult {
        if let Some(evaluator) = &mut self.evaluator {
            // evaluator.eval(runtime.adapted_actor());
        }
        HookResult::Continue
    }

    fn shutdown_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> Result<()> {
        runtime.shutdown();
        Ok(())
    }
}
