use std::io::Write;
use std::marker::PhantomData;
use std::{fs::File, path::PathBuf};

use r2l_core::ModeActorWrapper;
use r2l_core::on_policy::algorithm::{Agent, OnPolicyRuntime};
use r2l_core::{
    buffers::TrajectoryBatch,
    env::{Env, EnvBuilder, EnvBuilderType, normalizer::Normalizer},
    error::{Error, Result},
    models::{Actor, ToSafetensors},
    on_policy::algorithm::Sampler,
    tensor::R2lTensor,
};
use r2l_sampler::{DirectSampler, RolloutMode, SamplerExecutionMode, StagedSampler};
use yaml_serde::to_string;

use crate::{
    EpisodeBoundHook, TrainingLimit,
    constants::{ACTOR_FILE, EVALUATIONS_FILE, NORMALIZER_FILE},
    hooks::progress::TrainingProgress,
};

struct BestPolicyArtifacts<T: R2lTensor> {
    folder: PathBuf,
    best_reward: Option<f32>,
    normalizer: Option<Normalizer<T>>,
}

impl<T: R2lTensor> BestPolicyArtifacts<T> {
    fn write<A: ToSafetensors>(&mut self, actor: &A, avg_reward: f32) -> Result<()> {
        if self.best_reward.is_none_or(|r| avg_reward > r) {
            let bytes = actor.to_safetensors()?;
            let normalizer = if let Some(normalizer) = &self.normalizer {
                Some(to_string(&normalizer.snapshot()?).map_err(Error::wrap)?)
            } else {
                None
            };
            std::fs::write(self.folder.join(ACTOR_FILE), bytes).map_err(Error::wrap)?;
            if let Some(normalizer) = normalizer {
                std::fs::write(self.folder.join(NORMALIZER_FILE), normalizer)
                    .map_err(Error::wrap)?;
            }
            self.best_reward = Some(avg_reward);
        }
        Ok(())
    }
}

struct Artifacts<T: R2lTensor> {
    evaluation_results: Option<File>,
    best_policy: Option<BestPolicyArtifacts<T>>,
}

impl<T: R2lTensor> Artifacts<T> {
    fn new(
        folder: PathBuf,
        evaluation_results: bool,
        inference_artifacts: bool,
        normalizer: Option<Normalizer<T>>,
    ) -> Result<Self> {
        std::fs::create_dir_all(&folder).map_err(Error::wrap)?;
        let evaluation_results = if evaluation_results {
            let mut file = File::create(folder.join(EVALUATIONS_FILE)).map_err(Error::wrap)?;
            writeln!(file, "average_reward,total_episodes").map_err(Error::wrap)?;
            Some(file)
        } else {
            None
        };
        let best_policy = inference_artifacts.then_some(BestPolicyArtifacts {
            folder,
            best_reward: None,
            normalizer,
        });
        Ok(Self {
            evaluation_results,
            best_policy,
        })
    }

    fn write<A: ToSafetensors>(&mut self, actor: &A, avg_reward: f32, episodes: f32) -> Result<()> {
        if let Some(file) = &mut self.evaluation_results {
            writeln!(file, "{avg_reward},{episodes}").map_err(Error::wrap)?;
        }
        if let Some(best_policy) = &mut self.best_policy {
            best_policy.write(actor, avg_reward)?;
        }
        Ok(())
    }
}

pub(crate) enum EvaluationSampler<E: Env> {
    Direct(DirectSampler<E, EpisodeBoundHook<E>>),
    Staged(StagedSampler<E, EpisodeBoundHook<E>>),
}

impl<E: Env> EvaluationSampler<E> {
    pub(crate) fn build<EB: EnvBuilder<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        n_episodes: usize,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<Self> {
        let progress = TrainingProgress::shared(
            TrainingLimit::rollouts(1),
            RolloutMode::EpisodeBound { n_episodes },
            env_builder.num_envs(),
        );
        let hook = EpisodeBoundHook::new(progress, None);
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

    fn evaluate_with_sampler<S: Sampler<Tensor = E::Tensor>>(
        sampler: &mut S,
        actor: impl Actor<Tensor = E::Tensor> + Clone,
    ) -> Result<(f32, f32)> {
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

    fn evaluate<A: Actor<Tensor = E::Tensor> + Clone>(&mut self, actor: A) -> Result<(f32, f32)> {
        match self {
            Self::Direct(sampler) => Self::evaluate_with_sampler(sampler, actor),
            Self::Staged(sampler) => Self::evaluate_with_sampler(sampler, actor),
        }
    }

    fn normalizer(&self) -> Option<Normalizer<E::Tensor>> {
        match self {
            Self::Direct(_) => None,
            Self::Staged(sampler) => sampler.obs_normalizer().cloned(),
        }
    }
}

pub(crate) struct BestPolicyEvaluator<A: Actor, E: Env> {
    sampler: EvaluationSampler<E>,
    artifacts: Artifacts<E::Tensor>,
    _actor: PhantomData<A>,
}

impl<A: Actor + ToSafetensors + Clone, E: Env> BestPolicyEvaluator<A, E> {
    pub(crate) fn new(
        sampler: EvaluationSampler<E>,
        output_dir: PathBuf,
        evaluation_results: bool,
        inference_artifacts: bool,
    ) -> Result<Self> {
        let normalizer = sampler.normalizer();
        let artifacts = Artifacts::new(
            output_dir,
            evaluation_results,
            inference_artifacts,
            normalizer,
        )?;
        Ok(Self {
            sampler,
            artifacts,
            _actor: PhantomData,
        })
    }

    pub(crate) fn evaluate<AG: Agent<Actor = A>, TS: Sampler<Tensor = E::Tensor>>(
        &mut self,
        rt: &mut OnPolicyRuntime<AG, TS>,
    ) -> Result<()> {
        let actor = rt.actor();
        let adapted_actor = ModeActorWrapper::new(actor.clone());
        let (reward, episodes) = self.sampler.evaluate(adapted_actor)?;
        let avg_reward = reward / episodes;
        self.artifacts.write(&actor, avg_reward, episodes)?;
        Ok(())
    }

    pub(crate) fn finish_training(&self) -> Result<()> {
        if let Some(BestPolicyArtifacts { best_reward, .. }) = &self.artifacts.best_policy
            && best_reward.is_none()
        {
            Err(Error::InvalidState {
                operation: "serializing actor".into(),
                details: "no actor was evaluated, serialization is not possible".into(),
            })
        } else {
            Ok(())
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
    /// # Arguments
    ///
    /// * `episodes_per_evaluation` - Number of completed episodes collected per evaluation pass.
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
        assert!(
            rollouts_per_evaluation > 0,
            "rollouts per evaluation must be greater than zero"
        );
        self.rollouts_per_evaluation = rollouts_per_evaluation;
        self
    }
}
