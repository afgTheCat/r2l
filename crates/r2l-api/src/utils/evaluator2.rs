use std::path::PathBuf;

use anyhow::Result;
use r2l_core::{
    buffers::gen_buffer::TrajectoryBatchT,
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm2::Sampler2,
};
use r2l_sampler::{SamplerExecutionMode, sampler2::R2lSampler2};

use crate::hooks::sampler2::EpisodeBoundHook;

pub struct BestActorEvaluatorBuilder2<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
    n_episodes: usize,
    execution_mode: SamplerExecutionMode,
    eval_path: Option<PathBuf>,
}

impl<EB: EnvBuilder> BestActorEvaluatorBuilder2<EB> {
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

    pub fn with_best_actor_path<P: Into<PathBuf>>(mut self, eval_path: P) -> Self {
        self.eval_path = Some(eval_path.into());
        self
    }

    pub fn build<A: Actor>(self) -> BestActorEvaluator2<EB::Env, A> {
        let sampler = R2lSampler2::build(
            self.env_builder,
            EpisodeBoundHook::new(self.n_episodes),
            self.execution_mode,
        );
        BestActorEvaluator2 {
            sampler,
            best_actor_path: self.eval_path,
            best_rewards: f32::MIN,
            best_actor: None,
        }
    }
}

pub struct BestActorEvaluator2<E: Env, A: Actor> {
    sampler: R2lSampler2<E, EpisodeBoundHook<E>>,
    best_actor_path: Option<PathBuf>,
    best_actor: Option<A>,
    best_rewards: f32,
}

impl<E: Env, A: Actor> BestActorEvaluator2<E, A> {
    pub fn eval(&mut self, adapted_actor: impl Actor<Tensor = E::Tensor> + Clone, actor: A) {
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

    pub fn shutdown(&mut self) {
        self.sampler.shutdown();
    }
}
