use std::{marker::PhantomData, path::PathBuf};

use anyhow::Result;
use r2l_core::{
    buffers::{TrajectoryBatch, buffer::TrajectoryView},
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::{DefaultAdapter, OnPolicyAdapters, Sampler},
    tensor::R2lTensor,
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{R2lSampler, SamplerExecutionMode};

use crate::hooks::sampler::EpisodeBoundHook;

/// Builder for [`BestActorEvaluator`] instances.
pub struct BestActorEvaluatorBuilder<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
    n_episodes: usize,
    execution_mode: SamplerExecutionMode,
    eval_path: Option<PathBuf>,
    evaluator_frequency: usize,
}

impl<EB: EnvBuilder> BestActorEvaluatorBuilder<EB> {
    /// Creates an evaluator builder from an already-prepared environment builder type.
    pub fn from_env_builder_type(env_builder: EnvBuilderType<EB>) -> Self {
        Self {
            env_builder,
            evaluator_frequency: 1,
            n_episodes: 5,
            execution_mode: SamplerExecutionMode::Thread,
            eval_path: None,
        }
    }

    /// Creates an evaluator builder from a homogeneous environment builder.
    pub fn new(env_builder: EB) -> Self {
        Self {
            evaluator_frequency: 1,
            env_builder: EnvBuilderType::homogenous(env_builder.into(), 10),
            n_episodes: 5,
            execution_mode: SamplerExecutionMode::Thread,
            eval_path: None,
        }
    }

    /// Sets the frequency with which the evaluator runs
    pub fn with_evaluator_frequency(mut self, evaluator_frequency: usize) -> Self {
        self.evaluator_frequency = evaluator_frequency;
        self
    }

    /// Replaces the environment builder used for evaluation.
    pub fn with_env_builder(mut self, env_builder: EnvBuilderType<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

    /// Sets the number of episodes collected during each evaluation pass.
    pub fn with_n_episodes(mut self, n_episodes: usize) -> Self {
        self.n_episodes = n_episodes;
        self
    }

    /// Sets how evaluation workers are executed.
    pub fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    /// Sets the optional file path used to persist the best actor.
    pub fn with_best_actor_path<P: Into<PathBuf>>(mut self, eval_path: P) -> Self {
        self.eval_path = Some(eval_path.into());
        self
    }

    /// Builds a best-actor evaluator for the requested actor type.
    pub fn build<A: Actor>(self) -> BestActorEvaluator<EB::Env, A>
    where
        EB::Env: Env<Tensor: R2lTensor>,
    {
        let sampler = R2lSampler::build(
            self.env_builder,
            EpisodeBoundHook::new(self.n_episodes),
            self.execution_mode,
        );
        BestActorEvaluator {
            current_evaluator_step: 0,
            evaluator_frequency: self.evaluator_frequency,
            sampler,
            best_actor_path: self.eval_path,
            best_rewards: f32::MIN,
            best_actor: None,
        }
    }
}

/// Evaluates an actor through the sampler path and keeps the best one seen.
///
/// This evaluator collects episode-bounded rollouts with [`R2lSampler`],
/// computes the average completed-episode reward, and retains the best actor
/// observed so far.
pub struct BestActorEvaluator<E: Env<Tensor: R2lTensor>, A: Actor> {
    sampler: R2lSampler<E, EpisodeBoundHook<E>>,
    best_actor_path: Option<PathBuf>,
    best_actor: Option<A>,
    best_rewards: f32,
    current_evaluator_step: usize,
    evaluator_frequency: usize,
}

impl<E: Env<Tensor: R2lTensor>, A: Actor> BestActorEvaluator<E, A> {
    /// Evaluates the actor and stores it if it outperforms the current best actor.
    pub fn eval(&mut self, adapted_actor: impl Actor<Tensor = E::Tensor> + Clone, actor: A) {
        self.current_evaluator_step += 1;
        if self.current_evaluator_step % self.evaluator_frequency == 0 {
            self.sampler.reset_all_envs();
            self.sampler.collect_rollouts(adapted_actor);
            let trajectories = self.sampler.trajectory_views();
            let total_reward: f32 = trajectories
                .as_ref()
                .iter()
                .map(|x| x.rewards().iter().sum::<f32>())
                .sum();
            let total_episodes: f32 = trajectories
                .as_ref()
                .iter()
                .map(|b| b.episode_terminations() as f32)
                .sum();
            let avg_reward = total_reward / total_episodes;
            if avg_reward > self.best_rewards {
                self.best_rewards = avg_reward;
                self.best_actor = Some(actor);
            }
        }
    }

    /// Serializes and writes the current best actor to disk when supported.
    pub fn try_write_to_file(&self) -> Result<()> {
        let Some(actor) = self.best_actor.as_ref() else {
            return Ok(());
        };
        let Some(bytes) = actor.try_serialize() else {
            return Ok(());
        };
        let Some(path) = self.best_actor_path.as_ref() else {
            return Ok(());
        };
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Releases evaluator resources.
    pub fn shutdown(&mut self) {
        self.sampler.shutdown();
    }
}
