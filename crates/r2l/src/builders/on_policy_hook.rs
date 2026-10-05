use std::{num::NonZeroUsize, path::PathBuf};

use r2l_core::{
    env::{Env, normalizer::Normalizer},
    error::Error,
    models::{Actor, ToSafetensors},
    on_policy::algorithm::{Agent, Sampler},
};
use r2l_sampler::SamplerExecutionMode;

use crate::{
    EvaluationSettings, OnPolicyTrainingHooks, TrainingLimit,
    evaluator::{BestPolicyEvaluator, EvaluationSampler},
    hooks::{
        on_policy::{
            commands::{OnPolicyCommandHandler, OnPolicyControlEndpoint},
            evaluation::ScheduledEvaluator,
            timing::TimingRecorder,
        },
        progress::SharedTrainingProgress,
    },
};

trait EvaluatorSamplerBuiler<E: Env> {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: NonZeroUsize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<EvaluationSampler<E>, Error>;
}

pub struct TrainingArtifactsConfig<E: Env> {
    pub(crate) output_dir: PathBuf,
    pub(crate) evaluation_results: bool,
    pub(crate) training_timings: bool,
    pub(crate) inference_artifacts: bool,
    pub(crate) avg_reward_threshold: Option<f32>,
    pub(crate) evaluation_settings: EvaluationSettings,
    pub(crate) evaluator_sampler_builder: Box<dyn EvaluatorSamplerBuiler<E>>,
}

impl<E: Env> TrainingArtifactsConfig<E> {
    fn scheduled_evaluator<A: Actor + ToSafetensors + Clone>(
        &self,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
        progress: SharedTrainingProgress,
    ) -> ScheduledEvaluator<A, E> {
        let needs_evaluator = self.evaluation_results
            || self.inference_artifacts
            || self.avg_reward_threshold.is_some();
        if !needs_evaluator {
            return ScheduledEvaluator::Disabled;
        }
        let sampler = self
            .evaluator_sampler_builder
            .build_evaluator_sampler(
                self.evaluation_settings.episodes_per_evaluation,
                self.evaluation_settings.evaluation_execution_mode,
                obs_normalizer,
            )
            .unwrap();
        let evaluator = BestPolicyEvaluator::new(
            sampler,
            Some(self.output_dir.clone()),
            self.evaluation_results,
            self.inference_artifacts,
        )
        .unwrap();
        ScheduledEvaluator::new(
            evaluator,
            self.evaluation_settings.rollouts_per_evaluation,
            progress,
            self.avg_reward_threshold,
        )
    }

    fn timing_recorder(&self, progress: SharedTrainingProgress) -> TimingRecorder {
        if self.training_timings {
            TimingRecorder::create(&self.output_dir, progress).unwrap()
        } else {
            TimingRecorder::disabled()
        }
    }
}

pub struct OnPolicyHookConfig<E: Env> {
    training_limit: TrainingLimit,
    training_artifacts: TrainingArtifactsConfig<E>,
    control_endpoint: Option<OnPolicyControlEndpoint>,
}

impl<E: Env> OnPolicyHookConfig<E> {
    pub(crate) fn build<A: Agent<Actor: ToSafetensors>, S: Sampler<Tensor = E::Tensor>>(
        self,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
        progress: SharedTrainingProgress,
    ) -> OnPolicyTrainingHooks<A, S, E> {
        let evaluator = self
            .training_artifacts
            .scheduled_evaluator(obs_normalizer, progress.clone());
        let timing_recorder = self.training_artifacts.timing_recorder(progress.clone());
        let command_handler = OnPolicyCommandHandler::new(self.control_endpoint);
        OnPolicyTrainingHooks::new(progress, evaluator, command_handler, timing_recorder)
    }
}
