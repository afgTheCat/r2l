use std::{marker::PhantomData, path::PathBuf};

use anyhow::Result;
use r2l_core::{
    buffers::{TrajectoryContainer, variable_sized::VariableSizedStateBuffer},
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::{DefaultAdapter, OnPolicyAdapters, Sampler},
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{EpisodeTrajectoryBound, R2lSampler, SamplerExecutionMode};

pub struct BestActorEvaluatorBuilder<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
    n_episodes: usize,
    execution_mode: SamplerExecutionMode,
    eval_path: Option<PathBuf>,
}

impl<EB: EnvBuilder> BestActorEvaluatorBuilder<EB> {
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

    fn build<A: Actor>(self) -> BestActorEvaluator<EB::Env, A> {
        let sampler = R2lSampler::build(
            self.env_builder,
            EpisodeTrajectoryBound::new(self.n_episodes),
            self.execution_mode,
        );
        BestActorEvaluator {
            sampler,
            best_actor_path: self.eval_path,
            best_rewards: f32::MIN,
            best_actor: None,
        }
    }
}

/// Evaluates an actor and if it performs better than the previous one,
/// it saves backs it up. Optionally can also write it to a file.
pub struct BestActorEvaluator<E: Env, A: Actor> {
    sampler: R2lSampler<E, EpisodeTrajectoryBound<E::Tensor>>,
    best_actor_path: Option<PathBuf>,
    best_actor: Option<A>,
    best_rewards: f32,
}

impl<E: Env, A: Actor> BestActorEvaluator<E, A> {
    pub fn eval(&mut self, adapted_actor: impl Actor<Tensor = E::Tensor> + Clone, actor: A) {
        self.sampler.reset_all_envs();
        self.sampler.collect_rollouts(adapted_actor);
        let trajectories = self.sampler.trajectory_containers();
        let total_reward: f32 = trajectories
            .as_ref()
            .iter()
            .map(|x| x.rewards().sum::<f32>())
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

pub struct Evaluator<
    E: Env,
    A: Actor,
    AD: OnPolicyAdapters<A, R2lSampler<E, EpisodeTrajectoryBound<E::Tensor>>> = DefaultAdapter,
> {
    sampler: R2lSampler<E, EpisodeTrajectoryBound<E::Tensor>>,
    adapter: AD,
    _phantom: PhantomData<A>,
}

impl<E: Env, A: Actor> Evaluator<E, A, DefaultAdapter>
where
    DefaultAdapter: OnPolicyAdapters<A, R2lSampler<E, EpisodeTrajectoryBound<E::Tensor>>>,
{
    pub fn new<EB: EnvBuilder<Env = E>>(
        builder: EB,
        n_episodes: usize,
        n_env: usize,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        let rollout_bound = EpisodeTrajectoryBound::new(n_episodes);
        let env_builder = EnvBuilderType::homogenous(builder, n_env);
        let sampler = R2lSampler::build(env_builder, rollout_bound, execution_mode);
        Self {
            sampler,
            adapter: DefaultAdapter,
            _phantom: PhantomData,
        }
    }
}

impl<A: Actor> Evaluator<GymEnv, A, DefaultAdapter>
where
    DefaultAdapter:
        OnPolicyAdapters<A, R2lSampler<GymEnv, EpisodeTrajectoryBound<<GymEnv as Env>::Tensor>>>,
{
    pub fn gym<EB: Into<GymEnvBuilder>>(
        builder: EB,
        n_episodes: usize,
        n_env: usize,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        Self::new(builder.into(), n_episodes, n_env, execution_mode)
    }
}

impl<E: Env, A: Actor, AD: OnPolicyAdapters<A, R2lSampler<E, EpisodeTrajectoryBound<E::Tensor>>>>
    Evaluator<E, A, AD>
{
    pub fn eval(&mut self, actor: A) -> impl AsRef<[VariableSizedStateBuffer<E::Tensor>]> {
        let adapted_actor = self.adapter.adapt_actor(actor);
        self.sampler.reset_all_envs();
        self.sampler.collect_rollouts(adapted_actor);
        self.sampler.trajectory_containers()
    }
}
