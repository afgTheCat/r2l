use std::path::PathBuf;

use anyhow::Result;
use r2l_core::{
    buffers::TrajectoryBatch,
    env::{EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::{Agent, OnPolicyAdapters, OnPolicyRuntime, Sampler},
};
use r2l_sampler::{R2lSampler, SamplerExecutionMode};

use crate::hooks::sampler::EpisodeBoundHook;

struct EvalState {
    avg_reward: f32,
    total_episodes: f32,
}

/// Builder for [`BestActorEvaluator`] instances.
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
            execution_mode: SamplerExecutionMode::Thread,
            eval_path: None,
            csv_states_path: None,
            eval_states: vec![],
        }
    }

    /// Creates an evaluator builder from a homogeneous environment builder.
    pub fn new(env_builder: EB) -> Self {
        Self {
            evaluator_frequency: 1,
            env_builder: EnvBuilderType::homogenous(env_builder, 10),
            n_episodes: 5,
            execution_mode: SamplerExecutionMode::Thread,
            eval_path: None,
            csv_states_path: None,
            eval_states: vec![],
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

    /// Builds a best-actor evaluator for the requested actor type.
    pub fn build<A: Actor>(
        self,
    ) -> BestActorEvaluator<A, R2lSampler<EB::Env, EpisodeBoundHook<EB::Env>>> {
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
            csv_states_path: self.csv_states_path,
            eval_states: self.eval_states,
        }
    }

    /// Builds a best-actor evaluator around an already-constructed sampler.
    pub fn build_with_sampler<A: Actor, S: Sampler>(self, sampler: S) -> BestActorEvaluator<A, S> {
        BestActorEvaluator {
            current_evaluator_step: 0,
            evaluator_frequency: self.evaluator_frequency,
            sampler,
            best_actor_path: self.eval_path,
            best_rewards: f32::MIN,
            best_actor: None,
            csv_states_path: self.csv_states_path,
            eval_states: self.eval_states,
        }
    }

    pub(crate) fn n_episodes(&self) -> usize {
        self.n_episodes
    }

    pub(crate) fn execution_mode(&self) -> SamplerExecutionMode {
        self.execution_mode
    }

    pub(crate) fn env_builder(&self) -> &EnvBuilderType<EB> {
        &self.env_builder
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
/// This evaluator collects episode-bounded rollouts with [`R2lSampler`],
/// computes the average completed-episode reward, and retains the best actor
/// observed so far.
pub struct BestActorEvaluator<A: Actor, S: Sampler> {
    sampler: S,
    best_actor_path: Option<PathBuf>,
    best_actor: Option<A>,
    best_rewards: f32,
    current_evaluator_step: usize,
    evaluator_frequency: usize,
    csv_states_path: Option<PathBuf>,
    eval_states: Vec<EvalState>,
}

impl<A: Actor, ES: Sampler> BestActorEvaluator<A, ES> {
    pub fn eval<
        AG: Agent<Actor = A>,
        TS: Sampler<Tensor = ES::Tensor>,
        C: OnPolicyAdapters<AG::Actor, TS>,
    >(
        &mut self,
        rt: &mut OnPolicyRuntime<AG, TS, C>,
    ) {
        self.current_evaluator_step += 1;
        if self
            .current_evaluator_step
            .is_multiple_of(self.evaluator_frequency)
        {
            let actor = rt.actor();
            let adapted_actor = rt.adapted_actor();
            self.eval_adapted(adapted_actor, actor);
        }
    }

    /// Evaluates the actor and stores it if it outperforms the current best actor.
    pub fn eval_adapted(
        &mut self,
        adapted_actor: impl Actor<Tensor = ES::Tensor> + Clone,
        actor: A,
    ) {
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
        if self.csv_states_path.is_some() {
            self.eval_states.push(EvalState {
                avg_reward,
                total_episodes,
            });
        }
    }

    /// Serializes the current best actor and writes eval stats next to it.
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
        let Some(path) = self.csv_states_path.as_ref() else {
            return Ok(());
        };
        let mut csv = String::from("average_reward,total_episodes\n");
        for eval_state in &self.eval_states {
            csv.push_str(&format!(
                "{},{}\n",
                eval_state.avg_reward, eval_state.total_episodes
            ));
        }
        std::fs::write(path, csv)?;
        Ok(())
    }

    /// Releases evaluator resources.
    pub fn shutdown(&mut self) {
        self.sampler.shutdown();
    }
}
