use std::{fmt::Write as _, path::PathBuf};

use r2l_core::{
    ActorWrapper,
    buffers::TrajectoryBatch,
    env::{Env, EnvBuilder, EnvBuilderType, normalizer::ClippedNormalizer},
    error::Error,
    models::{Actor, ToSafetensors},
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
    tensor::R2lTensor,
};
use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};

use crate::{
    builders::{
        inference::{ACTOR_FILE, NORMALIZER_FILE},
        normalizer::NormalizerBuilder,
    },
    hooks::sampler::EpisodeBoundHook,
};

const EVALUATIONS_FILE: &str = "evaluations.csv";

pub(crate) enum EvaluationSampler<E: Env> {
    Direct(DirectSampler<E, EpisodeBoundHook<E>>),
    Staged(StagedSampler<E, EpisodeBoundHook<E>>),
}

impl<E: Env> EvaluationSampler<E> {
    pub(crate) fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        n_episodes: usize,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Result<Self, Error> {
        let hook = EpisodeBoundHook::new(n_episodes);
        if let Some(obs_normalizer) = obs_normalizer {
            Ok(Self::Staged(StagedSampler::build_with_obs_normalizer(
                &env_builder,
                hook,
                execution_mode,
                Some(obs_normalizer),
            )?))
        } else {
            Ok(Self::Direct(DirectSampler::build(
                env_builder,
                hook,
                execution_mode,
            )))
        }
    }

    fn evaluate<A: Actor<Tensor = E::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> Result<(f32, f32), Error> {
        match self {
            Self::Direct(sampler) => Self::evaluate_with_sampler(sampler, actor),
            Self::Staged(sampler) => Self::evaluate_with_sampler(sampler, actor),
        }
    }

    fn evaluate_with_sampler<S: Sampler<Tensor = E::Tensor>>(
        sampler: &mut S,
        actor: impl Actor<Tensor = E::Tensor> + Clone,
    ) -> Result<(f32, f32), Error> {
        sampler.reset_all_envs()?;
        sampler.collect_rollouts(actor)?;
        let trajectories = sampler.trajectory_views();
        let total_reward = trajectories
            .as_ref()
            .iter()
            .map(|trajectory| trajectory.rewards().iter().sum::<f32>())
            .sum();
        let total_episodes = trajectories
            .as_ref()
            .iter()
            .map(|trajectory| trajectory.episode_terminations() as f32)
            .sum();
        Ok((total_reward, total_episodes))
    }

    fn normalizer_snapshot(&self) -> Result<Option<NormalizerBuilder>, Error> {
        match self {
            Self::Direct(_) => Ok(None),
            Self::Staged(sampler) => sampler
                .obs_normalizer()
                .map(NormalizerBuilder::from_normalizer)
                .transpose()
                .map_err(Into::into),
        }
    }

    fn shutdown(&mut self) {
        match self {
            Self::Direct(sampler) => sampler.shutdown(),
            Self::Staged(sampler) => sampler.shutdown(),
        }
    }
}

/// Configures how policies are evaluated during training.
pub struct EvaluationSettings {
    pub(crate) episodes_per_evaluation: usize,
    pub(crate) evaluation_execution_mode: SamplerExecutionMode,
    pub(crate) rollouts_per_evaluation: usize,
}

impl Default for EvaluationSettings {
    fn default() -> Self {
        Self {
            rollouts_per_evaluation: 1,
            episodes_per_evaluation: 5,
            evaluation_execution_mode: SamplerExecutionMode::MultiThreaded,
        }
    }
}

impl EvaluationSettings {
    /// Creates evaluation settings with the default episode count, interval, and execution mode.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of episodes collected during each evaluation pass.
    ///
    /// # Panics
    ///
    /// Panics if `episodes_per_evaluation` is zero.
    #[must_use]
    pub fn with_episodes_per_evaluation(mut self, episodes_per_evaluation: usize) -> Self {
        assert!(
            episodes_per_evaluation > 0,
            "evaluation episode count must be greater than zero"
        );
        self.episodes_per_evaluation = episodes_per_evaluation;
        self
    }

    /// Sets how evaluation environments are executed.
    #[must_use]
    pub fn with_evaluation_execution_mode(
        mut self,
        evaluation_execution_mode: SamplerExecutionMode,
    ) -> Self {
        self.evaluation_execution_mode = evaluation_execution_mode;
        self
    }

    /// Sets the number of training rollouts between evaluation passes.
    ///
    /// # Panics
    ///
    /// Panics if `rollouts_per_evaluation` is zero.
    #[must_use]
    pub fn with_rollouts_per_evaluation(mut self, rollouts_per_evaluation: usize) -> Self {
        assert!(
            rollouts_per_evaluation > 0,
            "rollouts per evaluation must be greater than zero"
        );
        self.rollouts_per_evaluation = rollouts_per_evaluation;
        self
    }
}

struct EvalState {
    avg_reward: f32,
    total_episodes: f32,
}

/// Evaluates an actor through the sampler path and keeps the best one seen.
///
/// This evaluator collects episode-bounded rollouts,
/// computes the average completed-episode reward, and retains the best actor
/// observed so far.
pub(crate) struct BestActorEvaluator<A: Actor, E: Env> {
    sampler: EvaluationSampler<E>,
    output_dir: PathBuf,
    write_evaluation_results: bool,
    write_inference_artifacts: bool,
    best_actor: Option<A>,
    best_obs_normalizer: Option<NormalizerBuilder>,
    best_rewards: f32,
    eval_states: Vec<EvalState>,
}

impl<A: Actor + Clone + ToSafetensors, E: Env<Tensor: R2lTensor>> BestActorEvaluator<A, E> {
    pub(crate) fn new(
        sampler: EvaluationSampler<E>,
        output_dir: PathBuf,
        write_evaluation_results: bool,
        write_inference_artifacts: bool,
    ) -> Self {
        Self {
            sampler,
            output_dir,
            write_evaluation_results,
            write_inference_artifacts,
            best_actor: None,
            best_obs_normalizer: None,
            best_rewards: f32::MIN,
            eval_states: vec![],
        }
    }

    pub fn evaluate<AG: Agent<Actor = A>, TS: Sampler<Tensor = E::Tensor>>(
        &mut self,
        rt: &mut OnPolicyRuntime<AG, TS>,
    ) -> Result<(), Error> {
        let actor = rt.actor();
        let adapted_actor = ActorWrapper::new(rt.actor());
        self.eval_adapted(adapted_actor, actor)?;
        Ok(())
    }

    /// Evaluates the actor and persists it if it outperforms the current best actor.
    pub fn eval_adapted(
        &mut self,
        adapted_actor: impl Actor<Tensor = E::Tensor> + Clone,
        actor: A,
    ) -> Result<(), Error> {
        let (total_reward, total_episodes) = self.sampler.evaluate(adapted_actor)?;
        let avg_reward = total_reward / total_episodes;
        if self.write_evaluation_results {
            self.eval_states.push(EvalState {
                avg_reward,
                total_episodes,
            });
        }
        if avg_reward > self.best_rewards {
            self.best_rewards = avg_reward;
            if self.write_inference_artifacts {
                self.best_actor = Some(actor);
                self.best_obs_normalizer = self.sampler.normalizer_snapshot()?;
            }
            self.try_write_artifacts()?;
        }
        Ok(())
    }

    /// Writes the enabled inference artifacts and evaluation results.
    pub fn try_write_artifacts(&self) -> Result<(), Error> {
        std::fs::create_dir_all(&self.output_dir).map_err(Error::wrap)?;
        if self.write_inference_artifacts {
            let Some(actor) = &self.best_actor else {
                return Err(Error::InvalidState {
                    operation: "Serializing actor".into(),
                    details: "No actor was cached, serialization is not possible".into(),
                });
            };
            let bytes = actor.to_safetensors()?;
            std::fs::write(self.output_dir.join(ACTOR_FILE), bytes).map_err(Error::wrap)?;
            if let Some(normalizer) = &self.best_obs_normalizer {
                let normalizer_path = self.output_dir.join(NORMALIZER_FILE);
                let serialized = yaml_serde::to_string(normalizer).map_err(Error::wrap)?;
                std::fs::write(normalizer_path, serialized).map_err(Error::wrap)?;
            }
        }
        if self.write_evaluation_results {
            let mut csv = String::from("average_reward,total_episodes\n");
            for eval_state in &self.eval_states {
                writeln!(
                    csv,
                    "{},{}",
                    eval_state.avg_reward, eval_state.total_episodes
                )
                .map_err(Error::wrap)?;
            }
            std::fs::write(self.output_dir.join(EVALUATIONS_FILE), csv).map_err(Error::wrap)?;
        }
        Ok(())
    }

    /// Releases evaluator resources.
    pub fn shutdown(&mut self) {
        self.sampler.shutdown();
    }
}
