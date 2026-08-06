use std::path::PathBuf;

use anyhow::Result;
use r2l_core::{
    ActorWrapper,
    buffers::TrajectoryBatch,
    env::{Env, EnvBuilder, EnvBuilderType, normalizer::ClippedNormalizer},
    models::Actor,
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
    tensor::R2lTensor,
};
use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};
use serde::{Deserialize, Serialize};

use crate::{
    builders2::{
        inference::{ACTOR_FILE, NORMALIZER_FILE},
        normalizer::NormalizerBuilder,
    },
    hooks::sampler::EpisodeBoundHook,
};

fn resolve_and_validate_output_dir(path: PathBuf) -> PathBuf {
    let path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir().unwrap().join(path)
    };
    assert!(!path.is_file());
    path
}

const EVALUATIONS_FILE: &str = "evaluations.csv";

pub enum EvaluationSampler<E: Env> {
    Direct(DirectSampler<E, EpisodeBoundHook<E>>),
    Staged(StagedSampler<E, EpisodeBoundHook<E>>),
}

impl<E: Env> EvaluationSampler<E> {
    pub fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        n_episodes: usize,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Self {
        let hook = EpisodeBoundHook::new(n_episodes);
        if let Some(obs_normalizer) = obs_normalizer {
            Self::Staged(StagedSampler::build_with_obs_normalizer(
                env_builder,
                hook,
                execution_mode,
                Some(obs_normalizer),
            ))
        } else {
            Self::Direct(DirectSampler::build(env_builder, hook, execution_mode))
        }
    }

    fn evaluate<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, actor: A) -> (f32, f32) {
        match self {
            Self::Direct(sampler) => Self::evaluate_with_sampler(sampler, actor),
            Self::Staged(sampler) => Self::evaluate_with_sampler(sampler, actor),
        }
    }

    fn evaluate_with_sampler<S: Sampler<Tensor = E::Tensor>>(
        sampler: &mut S,
        actor: impl Actor<Tensor = E::Tensor> + Clone,
    ) -> (f32, f32) {
        sampler.reset_all_envs();
        sampler.collect_rollouts(actor);
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
        (total_reward, total_episodes)
    }

    fn normalizer_snapshot(&self) -> Option<NormalizerBuilder> {
        match self {
            Self::Direct(_) => None,
            Self::Staged(sampler) => sampler
                .core
                .obs_normalizer
                .clone()
                .map(NormalizerBuilder::from_normalizer),
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
#[derive(Serialize, Deserialize)]
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of episodes collected during each evaluation pass.
    pub fn with_episodes_per_evaluation(mut self, episodes_per_evaluation: usize) -> Self {
        assert!(
            episodes_per_evaluation > 0,
            "evaluation episode count must be greater than zero"
        );
        self.episodes_per_evaluation = episodes_per_evaluation;
        self
    }

    /// Sets how evaluation environments are executed.
    pub fn with_evaluation_execution_mode(
        mut self,
        evaluation_execution_mode: SamplerExecutionMode,
    ) -> Self {
        self.evaluation_execution_mode = evaluation_execution_mode;
        self
    }

    /// Sets the number of training rollouts between evaluation passes.
    pub fn with_rollouts_per_evaluation(mut self, rollouts_per_evaluation: usize) -> Self {
        assert!(
            rollouts_per_evaluation > 0,
            "rollouts per evaluation must be greater than zero"
        );
        self.rollouts_per_evaluation = rollouts_per_evaluation;
        self
    }
}

/// Selects the artifacts produced during training and where they are written.
#[derive(Serialize, Deserialize)]
pub struct TrainingArtifactsConfig {
    pub(crate) output_dir: PathBuf,
    pub(crate) evaluation_results: bool,
    pub(crate) performance_metrics: bool,
    pub(crate) inference_artifacts: bool,
    pub(crate) evaluation_settings: EvaluationSettings,
}

impl TrainingArtifactsConfig {
    /// Creates a configuration that writes all supported training artifacts.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: resolve_and_validate_output_dir(output_dir.into()),
            evaluation_results: true,
            performance_metrics: true,
            inference_artifacts: true,
            evaluation_settings: EvaluationSettings::default(),
        }
    }

    /// Sets whether evaluation results are written during training.
    pub fn with_evaluation_results(mut self, enabled: bool) -> Self {
        self.evaluation_results = enabled;
        self
    }

    /// Sets whether training performance metrics are written.
    pub fn with_performance_metrics(mut self, enabled: bool) -> Self {
        self.performance_metrics = enabled;
        self
    }

    /// Sets whether the best policy is saved as inference-ready artifacts.
    pub fn with_inference_artifacts(mut self, enabled: bool) -> Self {
        self.inference_artifacts = enabled;
        self
    }

    /// Sets the evaluation behavior used by evaluation results and inference artifacts.
    pub fn with_evaluation_settings(mut self, evaluation_settings: EvaluationSettings) -> Self {
        self.evaluation_settings = evaluation_settings;
        self
    }

    /// Builds an evaluator with an optional observation normalizer.
    pub fn build<A: Actor + Clone, EB: EnvBuilder>(
        self,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
        env_builder: EnvBuilderType<EB>,
    ) -> BestActorEvaluator<A, EB::Env> {
        let episodes_per_evaluation = self.evaluation_settings.episodes_per_evaluation;
        let evaluation_execution_mode = self.evaluation_settings.evaluation_execution_mode;
        let sampler = EvaluationSampler::build(
            env_builder,
            episodes_per_evaluation,
            evaluation_execution_mode,
            obs_normalizer,
        );
        self.build_with_sampler(sampler)
    }

    pub(crate) fn build_with_sampler<A: Actor + Clone, E: Env>(
        self,
        sampler: EvaluationSampler<E>,
    ) -> BestActorEvaluator<A, E> {
        BestActorEvaluator {
            current_evaluator_step: 0,
            rollouts_per_evaluation: self.evaluation_settings.rollouts_per_evaluation,
            sampler,
            output_dir: self.output_dir,
            write_evaluation_results: self.evaluation_results,
            write_inference_artifacts: self.inference_artifacts,
            best_rewards: f32::MIN,
            best_actor: None,
            best_obs_normalizer: None,
            eval_states: vec![],
        }
    }
}

#[derive(Serialize, Deserialize)]
struct EvalState {
    avg_reward: f32,
    total_episodes: f32,
}

/// Evaluates an actor through the sampler path and keeps the best one seen.
///
/// This evaluator collects episode-bounded rollouts,
/// computes the average completed-episode reward, and retains the best actor
/// observed so far.
pub struct BestActorEvaluator<A: Actor, E: Env> {
    sampler: EvaluationSampler<E>,
    output_dir: PathBuf,
    write_evaluation_results: bool,
    write_inference_artifacts: bool,
    best_actor: Option<A>,
    best_obs_normalizer: Option<NormalizerBuilder>,
    best_rewards: f32,
    current_evaluator_step: usize,
    rollouts_per_evaluation: usize,
    eval_states: Vec<EvalState>,
}

impl<A: Actor + Clone, E: Env<Tensor: R2lTensor>> BestActorEvaluator<A, E> {
    /// Evaluates the runtime actor when the configured interval elapses.
    /// Returns whether an evaluation was performed.
    pub fn eval<AG: Agent<Actor = A>, TS: Sampler<Tensor = E::Tensor>>(
        &mut self,
        rt: &mut OnPolicyRuntime<AG, TS>,
    ) -> bool {
        self.current_evaluator_step += 1;
        if self
            .current_evaluator_step
            .is_multiple_of(self.rollouts_per_evaluation)
        {
            let actor = rt.actor();
            let adapted_actor = ActorWrapper::new(rt.actor());
            self.eval_adapted(adapted_actor, actor);
            true
        } else {
            false
        }
    }

    /// Evaluates the actor and persists it if it outperforms the current best actor.
    pub fn eval_adapted(
        &mut self,
        adapted_actor: impl Actor<Tensor = E::Tensor> + Clone,
        actor: A,
    ) {
        let (total_reward, total_episodes) = self.sampler.evaluate(adapted_actor);
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
                self.best_obs_normalizer = self.sampler.normalizer_snapshot();
            }
            self.try_write_artifacts()
                .expect("failed to write training artifacts");
        }
    }

    /// Writes the enabled inference artifacts and evaluation results.
    pub fn try_write_artifacts(&self) -> Result<()> {
        std::fs::create_dir_all(&self.output_dir)?;
        if self.write_inference_artifacts
            && let Some(actor) = &self.best_actor
            && let Some(bytes) = actor.try_serialize()
        {
            std::fs::write(self.output_dir.join(ACTOR_FILE), bytes)?;
            if let Some(normalizer) = &self.best_obs_normalizer {
                let normalizer_path = self.output_dir.join(NORMALIZER_FILE);
                std::fs::write(normalizer_path, yaml_serde::to_string(normalizer)?)?;
            }
        }
        if self.write_evaluation_results {
            let mut csv = String::from("average_reward,total_episodes\n");
            for eval_state in &self.eval_states {
                csv.push_str(&format!(
                    "{},{}\n",
                    eval_state.avg_reward, eval_state.total_episodes
                ));
            }
            std::fs::write(self.output_dir.join(EVALUATIONS_FILE), csv)?;
        }
        Ok(())
    }

    /// Releases evaluator resources.
    pub fn shutdown(&mut self) {
        self.sampler.shutdown();
    }
}
