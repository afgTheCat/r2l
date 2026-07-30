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

use crate::{builders::normalizer::NormalizerBuilder, hooks::sampler::EpisodeBoundHook};

enum EvaluationSampler<E: Env> {
    Direct(DirectSampler<E, EpisodeBoundHook<E>>),
    Staged(StagedSampler<E, EpisodeBoundHook<E>>),
}

impl<E: Env> EvaluationSampler<E> {
    fn build<EB: EnvBuilder<Env = E>>(
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

#[derive(Serialize, Deserialize)]
struct EvalState {
    avg_reward: f32,
    total_episodes: f32,
}

/// Builder for [`BestActorEvaluator`] instances.
#[derive(Serialize, Deserialize)]
pub struct BestActorEvaluatorBuilder<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
    n_episodes: usize,
    execution_mode: SamplerExecutionMode,
    eval_path: Option<PathBuf>,
    evaluator_frequency: usize,
    csv_states_path: Option<PathBuf>,
    eval_states: Vec<EvalState>,
}

impl<EB: EnvBuilder> BestActorEvaluatorBuilder<EB> {
    /// Creates an evaluator builder from an already-prepared environment builder type.
    pub fn from_env_builder_type(env_builder: EnvBuilderType<EB>) -> Self {
        Self {
            env_builder,
            evaluator_frequency: 1,
            n_episodes: 5,
            execution_mode: SamplerExecutionMode::MultiThreaded,
            eval_path: None,
            csv_states_path: None,
            eval_states: vec![],
        }
    }

    /// Creates an evaluator builder from a homogeneous environment builder.
    pub fn new(env_builder: EB) -> Self {
        Self {
            evaluator_frequency: 1,
            env_builder: EnvBuilderType::homogeneous(env_builder, 10),
            n_episodes: 5,
            execution_mode: SamplerExecutionMode::MultiThreaded,
            eval_path: None,
            csv_states_path: None,
            eval_states: vec![],
        }
    }

    /// Sets the frequency with which the evaluator runs.
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
        let eval_path = assert_file_path_is_valid(eval_path.into());
        self.eval_path = Some(eval_path);
        self
    }

    /// Sets the optional CSV path used to persist evaluation states.
    pub fn with_csv_states<P: Into<PathBuf>>(mut self, csv_states_path: P) -> Self {
        let csv_states_path = assert_file_path_is_valid(csv_states_path.into());
        self.csv_states_path = Some(csv_states_path);
        self
    }

    /// Builds an evaluator with an optional observation normalizer.
    pub fn build<A: Actor + Clone>(
        self,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
    ) -> BestActorEvaluator<A, EB::Env> {
        let sampler = EvaluationSampler::build(
            self.env_builder,
            self.n_episodes,
            self.execution_mode,
            obs_normalizer,
        );
        BestActorEvaluator {
            current_evaluator_step: 0,
            evaluator_frequency: self.evaluator_frequency,
            sampler,
            best_actor_path: self.eval_path,
            best_rewards: f32::MIN,
            best_actor: None,
            best_obs_normalizer: None,
            csv_states_path: self.csv_states_path,
            eval_states: self.eval_states,
        }
    }
}

fn assert_file_path_is_valid(path: PathBuf) -> PathBuf {
    let path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir().unwrap().join(path)
    };
    let Some(parent) = path.parent() else {
        panic!("Path has to have a parent existing");
    };
    assert!(parent.is_dir());
    assert!(!path.is_dir());
    path
}

/// Evaluates an actor through the sampler path and keeps the best one seen.
///
/// This evaluator collects episode-bounded rollouts,
/// computes the average completed-episode reward, and retains the best actor
/// observed so far.
pub struct BestActorEvaluator<A: Actor, E: Env<Tensor: R2lTensor>> {
    sampler: EvaluationSampler<E>,
    best_actor_path: Option<PathBuf>,
    best_actor: Option<A>,
    best_obs_normalizer: Option<NormalizerBuilder>,
    best_rewards: f32,
    current_evaluator_step: usize,
    evaluator_frequency: usize,
    csv_states_path: Option<PathBuf>,
    eval_states: Vec<EvalState>,
}

impl<A: Actor + Clone, E: Env<Tensor: R2lTensor>> BestActorEvaluator<A, E> {
    /// Evaluates the runtime actor when the configured evaluation interval elapses.
    pub fn eval<AG: Agent<Actor = A>, TS: Sampler<Tensor = E::Tensor>>(
        &mut self,
        rt: &mut OnPolicyRuntime<AG, TS>,
    ) {
        self.current_evaluator_step += 1;
        if self
            .current_evaluator_step
            .is_multiple_of(self.evaluator_frequency)
        {
            let actor = rt.actor();
            let adapted_actor = ActorWrapper::new(rt.actor());
            self.eval_adapted(adapted_actor, actor);
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
        if self.csv_states_path.is_some() {
            self.eval_states.push(EvalState {
                avg_reward,
                total_episodes,
            });
        }
        if avg_reward > self.best_rewards {
            self.best_rewards = avg_reward;
            self.best_actor = Some(actor);
            self.best_obs_normalizer = self.sampler.normalizer_snapshot();
            self.try_write_to_file()
                .expect("failed to write improved actor checkpoint");
        }
    }

    /// Serializes the current best actor and writes eval stats next to it.
    pub fn try_write_to_file(&self) -> Result<()> {
        if let (Some(actor), Some(path)) = (&self.best_actor, &self.best_actor_path)
            && let Some(bytes) = actor.try_serialize()
        {
            std::fs::write(path, bytes)?;
            if let Some(normalizer) = &self.best_obs_normalizer {
                let normalizer_path = path.with_extension("normalizer.yaml");
                std::fs::write(normalizer_path, yaml_serde::to_string(normalizer)?)?;
            }
        }
        if let Some(path) = &self.csv_states_path {
            let mut csv = String::from("average_reward,total_episodes\n");
            for eval_state in &self.eval_states {
                csv.push_str(&format!(
                    "{},{}\n",
                    eval_state.avg_reward, eval_state.total_episodes
                ));
            }
            std::fs::write(path, csv)?;
        }
        Ok(())
    }

    /// Releases evaluator resources.
    pub fn shutdown(&mut self) {
        self.sampler.shutdown();
    }
}
