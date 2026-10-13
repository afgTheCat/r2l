use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use r2l_core::ModeActorWrapper;
use r2l_core::on_policy::algorithm::{Agent, OnPolicyRuntime};
use r2l_core::{
    buffers::TrajectoryBatch,
    env::{Env, EnvBuilderType, normalizer::Normalizer},
    error::{Error, Result},
    models::{Actor, ToSafetensors},
    on_policy::algorithm::Sampler,
    tensor::R2lTensor,
};
use r2l_sampler::{RolloutMode, SamplerExecutionMode};
use serde::{Deserialize, Serialize};
use yaml_serde::to_string;

use crate::{
    OnPolicySampler, SamplerHook,
    constants::{ACTOR_FILE, EVALUATIONS_FILE, NORMALIZER_FILE},
    csv_artifact::CsvArtifact,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct EvaluationResult {
    average_reward: f32,
    #[serde(rename = "total_episodes")]
    num_episodes: NonZeroUsize,
}

impl EvaluationResult {
    pub(crate) fn new(total_reward: f32, num_episodes: usize) -> Self {
        let num_episodes = NonZeroUsize::new(num_episodes).expect("Results cannot be empty");
        Self {
            average_reward: total_reward / num_episodes.get() as f32,
            num_episodes,
        }
    }

    pub(crate) fn avg_reward(&self) -> f32 {
        self.average_reward
    }
}

struct BestPolicyArtifacts<T: R2lTensor> {
    best_reward: Option<f32>,
    normalizer: Option<Normalizer<T>>,
}

impl<T: R2lTensor> BestPolicyArtifacts<T> {
    fn new(normalizer: Option<Normalizer<T>>) -> Self {
        Self {
            best_reward: None,
            normalizer,
        }
    }

    fn write<A: ToSafetensors>(&mut self, folder: &Path, actor: &A, avg_reward: f32) -> Result<()> {
        if self.best_reward.is_none_or(|r| avg_reward > r) {
            let bytes = actor.to_safetensors()?;
            let normalizer = if let Some(normalizer) = &self.normalizer {
                Some(to_string(&normalizer.snapshot()?).map_err(Error::wrap)?)
            } else {
                None
            };
            std::fs::write(folder.join(ACTOR_FILE), bytes)?;
            if let Some(normalizer) = normalizer {
                std::fs::write(folder.join(NORMALIZER_FILE), normalizer)?;
            }
            self.best_reward = Some(avg_reward);
        }
        Ok(())
    }
}

struct Artifacts<T: R2lTensor> {
    folder: PathBuf,
    evaluation_results: Option<CsvArtifact<EvaluationResult>>,
    best_policy: Option<BestPolicyArtifacts<T>>,
}

impl<T: R2lTensor> Artifacts<T> {
    fn new(
        folder: PathBuf,
        evaluation_results: bool,
        inference_artifacts: bool,
        normalizer: Option<Normalizer<T>>,
    ) -> Result<Self> {
        std::fs::create_dir_all(&folder)?;
        let evaluation_results = evaluation_results
            .then(|| CsvArtifact::open(&folder.join(EVALUATIONS_FILE)))
            .transpose()?;
        let best_policy = inference_artifacts.then_some(BestPolicyArtifacts::new(normalizer));
        Ok(Self {
            folder,
            evaluation_results,
            best_policy,
        })
    }

    fn write<A: ToSafetensors>(&mut self, actor: &A, eval_result: EvaluationResult) -> Result<()> {
        let avg_reward = eval_result.avg_reward();
        if let Some(file) = &mut self.evaluation_results {
            file.write(&eval_result)?;
        }
        if let Some(best_policy) = &mut self.best_policy {
            best_policy.write(&self.folder, actor, avg_reward)?;
        }
        Ok(())
    }
}

pub(crate) struct EvaluationSampler<E: Env + 'static> {
    sampler: OnPolicySampler<E>,
}

impl<E: Env> EvaluationSampler<E> {
    pub(crate) fn build(
        env_builder: EnvBuilderType<E>,
        n_episodes: NonZeroUsize,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<Self> {
        let hook = SamplerHook::new(RolloutMode::EpisodeBound { n_episodes }, None);
        let sampler = match obs_normalizer {
            Some(normalizer) => {
                OnPolicySampler::build_staged(&env_builder, hook, execution_mode, normalizer)?
            }
            None => OnPolicySampler::build_direct(&env_builder, hook, execution_mode)?,
        };
        Ok(Self { sampler })
    }

    fn evaluate<A: Actor<Tensor = E::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> Result<EvaluationResult> {
        self.sampler.reset_all_envs()?;
        self.sampler.collect_rollouts(actor)?;
        let trajectories = self.sampler.trajectory_views();
        let total_reward = trajectories
            .as_ref()
            .iter()
            .map(|trajectory| trajectory.rewards().iter().sum::<f32>())
            .sum();
        let num_episodes: usize = trajectories
            .as_ref()
            .iter()
            .map(|trajectory| trajectory.episode_terminations())
            .sum();
        Ok(EvaluationResult::new(total_reward, num_episodes))
    }

    fn normalizer(&self) -> Option<Normalizer<E::Tensor>> {
        self.sampler.obs_normalizer()
    }
}

pub(crate) struct BestPolicyEvaluator<E: Env + 'static> {
    sampler: EvaluationSampler<E>,
    artifacts: Option<Artifacts<E::Tensor>>,
}

impl<E: Env> BestPolicyEvaluator<E> {
    pub(crate) fn best_reward(&self) -> Option<f32> {
        self.artifacts.as_ref()?.best_policy.as_ref()?.best_reward
    }

    pub(crate) fn restore_best_reward(&mut self, best_reward: Option<f32>) {
        if let Some(best_policy) = self.artifacts.as_mut().and_then(|a| a.best_policy.as_mut()) {
            best_policy.best_reward = best_reward;
        }
    }

    pub(crate) fn new(
        sampler: EvaluationSampler<E>,
        output_dir: Option<PathBuf>,
        evaluation_results: bool,
        inference_artifacts: bool,
    ) -> Result<Self> {
        let artifacts = output_dir
            .map(|output_dir| {
                Artifacts::new(
                    output_dir,
                    evaluation_results,
                    inference_artifacts,
                    sampler.normalizer(),
                )
            })
            .transpose()?;
        Ok(Self { sampler, artifacts })
    }

    pub(crate) fn evaluate<AG: Agent<Actor: ToSafetensors>, TS: Sampler<Tensor = E::Tensor>>(
        &mut self,
        rt: &mut OnPolicyRuntime<AG, TS>,
    ) -> Result<EvaluationResult> {
        let actor = rt.actor();
        let adapted_actor = ModeActorWrapper::new(actor.clone());
        let evaluation_result = self.sampler.evaluate(adapted_actor)?;
        if let Some(artifacts) = &mut self.artifacts {
            artifacts.write(&actor, evaluation_result)?;
        }
        Ok(evaluation_result)
    }
}

/// Configures how policies are evaluated during training.
#[derive(Serialize, Deserialize)]
pub struct EvaluationSettings {
    pub(crate) episodes_per_evaluation: NonZeroUsize,
    pub(crate) evaluation_execution_mode: SamplerExecutionMode,
    pub(crate) rollouts_per_evaluation: NonZeroUsize,
}

impl Default for EvaluationSettings {
    fn default() -> Self {
        Self {
            rollouts_per_evaluation: NonZeroUsize::new(1).unwrap(),
            episodes_per_evaluation: NonZeroUsize::new(5).unwrap(),
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
    /// # Arguments
    ///
    /// * `episodes_per_evaluation` - Number of completed episodes collected per evaluation pass.
    ///
    /// # Panics
    ///
    /// Panics if `episodes_per_evaluation` is zero.
    #[must_use]
    pub fn with_episodes_per_evaluation(mut self, episodes_per_evaluation: usize) -> Self {
        self.episodes_per_evaluation = NonZeroUsize::new(episodes_per_evaluation)
            .expect("evaluation episode count must be greater than zero");
        self
    }

    /// Sets how evaluation environments are executed.
    ///
    /// # Arguments
    ///
    /// * `evaluation_execution_mode` - Whether evaluation workers run on the current thread or on
    ///   dedicated threads.
    #[must_use]
    pub fn with_execution_mode(mut self, evaluation_execution_mode: SamplerExecutionMode) -> Self {
        self.evaluation_execution_mode = evaluation_execution_mode;
        self
    }

    /// Sets the number of training rollouts between evaluation passes.
    ///
    /// # Arguments
    ///
    /// * `rollouts_per_evaluation` - Number of completed training rollouts between evaluations.
    ///
    /// # Panics
    ///
    /// Panics if `rollouts_per_evaluation` is zero.
    #[must_use]
    pub fn with_rollouts_per_evaluation(mut self, rollouts_per_evaluation: usize) -> Self {
        self.rollouts_per_evaluation = NonZeroUsize::new(rollouts_per_evaluation)
            .expect("rollouts per evaluation must be greater than zero");
        self
    }
}
