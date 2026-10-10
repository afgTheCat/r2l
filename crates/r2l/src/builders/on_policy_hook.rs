use std::path::PathBuf;

use r2l_core::{
    env::{Env, normalizer::NormalizerMode},
    error::Error,
    models::ToSafetensors,
    on_policy::algorithm::{Agent, Sampler},
};

use super::{policy::PolicyBuilder, sampler::SamplerConfiguration};
use crate::{
    EvaluationSettings, OnPolicyTrainingHooks,
    backend::Backend,
    evaluator::BestPolicyEvaluator,
    hooks::{
        on_policy::{
            commands::{OnPolicyCommandHandler, OnPolicyControlEndpoint},
            evaluation::ScheduledEvaluator,
            timing::TimingRecorder,
        },
        progress::SharedTrainingProgress,
    },
    inference::{InferenceConfig, InferenceObservationMode},
};

/// Selects the artifacts produced during training and where they are written.
pub struct TrainingArtifactsConfig {
    pub(crate) output_dir: PathBuf,
    pub(crate) evaluation_results: bool,
    pub(crate) training_timings: bool,
    pub(crate) inference_artifacts: bool,
    pub(crate) evaluation_settings: EvaluationSettings,
}

impl TrainingArtifactsConfig {
    /// Creates a configuration that writes all supported training artifacts.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory in which training artifacts will be written.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: Self::resolve_and_validate_output_dir(output_dir.into()),
            evaluation_results: true,
            training_timings: true,
            inference_artifacts: true,
            evaluation_settings: EvaluationSettings::default(),
        }
    }

    /// Sets whether evaluation results are written during training.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to write evaluation results.
    #[must_use]
    pub fn with_evaluation_results(mut self, enabled: bool) -> Self {
        self.evaluation_results = enabled;
        self
    }

    /// Sets whether training timing measurements are written.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to write training timing measurements.
    #[must_use]
    pub fn with_training_timings(mut self, enabled: bool) -> Self {
        self.training_timings = enabled;
        self
    }

    /// Sets whether the best policy is saved as inference-ready artifacts.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to save inference-ready artifacts for the best policy.
    #[must_use]
    pub fn with_inference_artifacts(mut self, enabled: bool) -> Self {
        self.inference_artifacts = enabled;
        self
    }

    /// Sets the evaluation behavior used by artifacts and average-reward stopping.
    ///
    /// # Arguments
    ///
    /// * `evaluation_settings` - Settings that control evaluation frequency and execution.
    #[must_use]
    pub fn with_evaluation_settings(mut self, evaluation_settings: EvaluationSettings) -> Self {
        self.evaluation_settings = evaluation_settings;
        self
    }

    fn needs_evaluator(&self) -> bool {
        self.evaluation_results || self.inference_artifacts
    }

    fn resolve_and_validate_output_dir(path: PathBuf) -> PathBuf {
        let path = if path.is_absolute() {
            path
        } else {
            std::env::current_dir().unwrap().join(path)
        };
        assert!(!path.is_file());
        path
    }
}

/// Evaluation, artifact, and external-control settings for training lifecycle hooks.
#[derive(Default)]
pub struct OnPolicyHookConfig {
    pub(crate) avg_reward_threshold: Option<f32>,
    pub(crate) training_artifacts: Option<TrainingArtifactsConfig>,
    pub(crate) control_endpoint: Option<OnPolicyControlEndpoint>,
}

impl OnPolicyHookConfig {
    fn needs_evaluator(&self) -> bool {
        self.avg_reward_threshold.is_some()
            || self
                .training_artifacts
                .as_ref()
                .is_some_and(TrainingArtifactsConfig::needs_evaluator)
    }

    pub(crate) fn build<A: Agent<Actor: ToSafetensors>, S: Sampler<Tensor = E::Tensor>, E: Env>(
        self,
        sampler: &SamplerConfiguration<E>,
        policy: &PolicyBuilder,
        backend: &Backend,
        progress: SharedTrainingProgress,
    ) -> Result<OnPolicyTrainingHooks<A, S, E>, Error> {
        let obs_normalizer = sampler.obs_normalizer();
        if let Some(config) = &self.training_artifacts
            && config.inference_artifacts
        {
            let observation_mode = if obs_normalizer.is_some() {
                InferenceObservationMode::Normalized
            } else {
                InferenceObservationMode::Raw
            };
            InferenceConfig::new(policy.clone(), observation_mode, backend.clone())
                .write_to_dir(&config.output_dir)?;
        }
        let evaluator = if self.needs_evaluator() {
            let defaults = EvaluationSettings::default();
            let settings = self
                .training_artifacts
                .as_ref()
                .map_or(&defaults, |config| &config.evaluation_settings);
            let obs_normalizer = obs_normalizer.map(|n| n.with_mode(NormalizerMode::ReadOnly));
            let sampler = sampler.env_build_plan.build_evaluator_sampler(
                settings.episodes_per_evaluation,
                settings.evaluation_execution_mode,
                obs_normalizer,
            )?;
            let artifacts = self
                .training_artifacts
                .as_ref()
                .filter(|config| config.needs_evaluator());
            ScheduledEvaluator::new(
                BestPolicyEvaluator::new(
                    sampler,
                    artifacts.map(|config| config.output_dir.clone()),
                    artifacts.is_some_and(|config| config.evaluation_results),
                    artifacts.is_some_and(|config| config.inference_artifacts),
                )?,
                settings.rollouts_per_evaluation,
                progress.clone(),
                self.avg_reward_threshold,
            )
        } else {
            ScheduledEvaluator::disabled()
        };
        let timing_recorder = if let Some(config) = &self.training_artifacts
            && config.training_timings
        {
            TimingRecorder::create(&config.output_dir, progress.clone())?
        } else {
            TimingRecorder::disabled()
        };
        Ok(OnPolicyTrainingHooks::new(
            progress,
            evaluator,
            OnPolicyCommandHandler::new(self.control_endpoint),
            timing_recorder,
        ))
    }
}
