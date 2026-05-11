use std::{marker::PhantomData, path::PathBuf};

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
use r2l_sampler::{EpisodeTrajectoryBound, R2lSampler, SamplerExecutionMode};

pub struct EvaluatorBuilder<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
    n_episodes: usize,
    execution_mode: SamplerExecutionMode,
    eval_path: Option<PathBuf>,
}

impl<EB: EnvBuilder> EvaluatorBuilder<EB> {
    pub fn from_env_builder_type(env_builder: EnvBuilderType<EB>) -> Self {
        Self {
            env_builder,
            n_episodes: 5,
            execution_mode: SamplerExecutionMode::Thread,
            eval_path: None,
        }
    }

    pub fn new(env_builder: EB) -> Self {
        Self {
            env_builder: EnvBuilderType::homogenous(env_builder.into(), 10),
            n_episodes: 5,
            execution_mode: SamplerExecutionMode::Thread,
            eval_path: None,
        }
    }

    pub fn with_env_builder(mut self, env_builder: EnvBuilderType<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

    pub fn with_n_episodes(mut self, n_episodes: usize) -> Self {
        self.n_episodes = n_episodes;
        self
    }

    pub fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    pub fn with_eval_path<P: Into<PathBuf>>(mut self, eval_path: P) -> Self {
        self.eval_path = Some(eval_path.into());
        self
    }

    fn build<A: Actor>(self) -> Evaluator<EB::Env, A> {
        let sampler = R2lSampler::build(
            self.env_builder,
            EpisodeTrajectoryBound::new(self.n_episodes),
            self.execution_mode,
        );
        Evaluator {
            sampler,
            path: self.eval_path,
            best_rewards: f32::MIN,
            best_actor: None,
        }
    }
}

struct Evaluator<E: Env, A: Actor> {
    sampler: R2lSampler<E, EpisodeTrajectoryBound<E::Tensor>>,
    path: Option<PathBuf>,
    best_actor: Option<A>,
    best_rewards: f32,
}

impl<E: Env, A: Actor> Evaluator<E, A> {
    fn eval(&mut self, adapted_actor: impl Actor<Tensor = E::Tensor> + Clone, actor: A) {
        self.sampler.reset_all_envs();
        self.sampler.collect_rollouts(adapted_actor);
        let trajectories = self.sampler.trajectory_containers();
        let total_reward: f32 = trajectories
            .as_ref()
            .iter()
            .map(|x| x.rewards().sum::<f32>())
            .sum();
        let avg_reward = total_reward / trajectories.as_ref().len() as f32;
        println!("avg reward: {avg_reward}");
        if avg_reward > self.best_rewards {
            self.best_rewards = avg_reward;
            self.best_actor = Some(actor);
        }
    }

    fn try_write_to_file(&self) -> Result<()> {
        let Some(actor) = self.best_actor.as_ref() else {
            return Ok(());
        };
        let Some(bytes) = actor.try_serialize() else {
            return Ok(());
        };
        let Some(path) = self.path.as_ref() else {
            return Ok(());
        };
        std::fs::write(path, bytes)?;
        Ok(())
    }

    fn shutdown(&mut self) {
        self.sampler.shutdown();
    }
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
    learning_schedule: LearningSchedule,
    evaluator: Option<Evaluator<E, A::Actor>>,
    _phantom: PhantomData<(A, S, C)>,
}

impl<A: Agent, S: Sampler, C: OnPolicyAdapters<A, S>, E: Env<Tensor = S::Tensor>>
    DefaultOnPolicyAlgorithmHooks<A, S, C, E>
{
    /// Creates the default outer-loop hooks for the given learning schedule.
    pub fn new<EB: EnvBuilder<Env = E>>(
        learning_schedule: LearningSchedule,
        evaluator_builder: Option<EvaluatorBuilder<EB>>,
    ) -> Self {
        Self {
            learning_schedule,
            evaluator: evaluator_builder.map(EvaluatorBuilder::build),
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
        _runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult {
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
