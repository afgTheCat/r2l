use std::marker::PhantomData;

use r2l_core::{
    buffers::buffer::TrajectoryView,
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::{DefaultAdapter, OnPolicyAdapters, Sampler},
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{R2lSampler, SamplerExecutionMode};

use crate::hooks::sampler::EpisodeBoundHook;

/// Generic evaluation helper for the sampler/adapter path.
///
/// This helper adapts an actor to the sampler tensor type, collects
/// episode-bounded rollouts through [`R2lSampler`], and returns the resulting
/// trajectory views for inspection.
pub struct Evaluator<
    E: Env,
    A: Actor,
    AD: OnPolicyAdapters<A, R2lSampler<E, EpisodeBoundHook<E>, A::State>> = DefaultAdapter,
> {
    sampler: R2lSampler<E, EpisodeBoundHook<E>, A::State>,
    adapter: AD,
    _phantom: PhantomData<A>,
}

impl<E: Env, A: Actor> Evaluator<E, A, DefaultAdapter>
where
    DefaultAdapter: OnPolicyAdapters<A, R2lSampler<E, EpisodeBoundHook<E>, A::State>>,
{
    /// Creates a new evaluator for a custom environment builder.
    pub fn new<EB: EnvBuilder<Env = E>>(
        builder: EB,
        n_episodes: usize,
        n_env: usize,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        let hook = EpisodeBoundHook::new(n_episodes);
        let env_builder = EnvBuilderType::homogenous(builder, n_env);
        let sampler = R2lSampler::build(env_builder, hook, execution_mode);
        Self {
            sampler,
            adapter: DefaultAdapter,
            _phantom: PhantomData,
        }
    }
}

impl<A: Actor> Evaluator<GymEnv, A, DefaultAdapter>
where
    DefaultAdapter: OnPolicyAdapters<A, R2lSampler<GymEnv, EpisodeBoundHook<GymEnv>, A::State>>,
{
    /// Creates a new evaluator for a Gym environment.
    pub fn gym<EB: Into<GymEnvBuilder>>(
        builder: EB,
        n_episodes: usize,
        n_env: usize,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        Self::new(builder.into(), n_episodes, n_env, execution_mode)
    }
}

impl<E: Env, A: Actor, AD: OnPolicyAdapters<A, R2lSampler<E, EpisodeBoundHook<E>, A::State>>>
    Evaluator<E, A, AD>
{
    /// Evaluates an actor and returns the collected trajectory views.
    pub fn eval(
        &mut self,
        actor: A,
    ) -> impl AsRef<[TrajectoryView<'_, <<AD as OnPolicyAdapters<A, R2lSampler<E, EpisodeBoundHook<E>, A::State>>>::SamplerActor as Actor>::Tensor, A::State>]>
    {
        let adapted_actor = self.adapter.adapt_actor(actor);
        self.sampler.reset_all_envs();
        self.sampler.collect_rollouts(adapted_actor);
        self.sampler.trajectory_views()
    }
}
