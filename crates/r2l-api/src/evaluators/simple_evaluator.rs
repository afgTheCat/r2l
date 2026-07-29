use std::marker::PhantomData;

use r2l_core::{
    ActorWrapper,
    buffers::buffer::TrajectoryView,
    env::{Env, EnvBuilder, EnvBuilderType},
    models::Actor,
    on_policy::algorithm::Sampler,
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{DirectSampler, SamplerExecutionMode};

use crate::hooks::sampler::EpisodeBoundHook;

/// Generic evaluation helper using the standard sampler tensor conversion.
///
/// This helper adapts an actor to the sampler tensor type, collects
/// episode-bounded rollouts through [`DirectSampler`], and returns the resulting
/// trajectory views for inspection.
pub struct Evaluator<E: Env, A: Actor> {
    sampler: DirectSampler<E, EpisodeBoundHook<E>>,
    _phantom: PhantomData<A>,
}

impl<E: Env, A: Actor + Clone> Evaluator<E, A> {
    /// Creates a new evaluator for a custom environment builder.
    pub fn new<EB: EnvBuilder<Env = E>>(
        builder: EB,
        n_episodes: usize,
        n_env: usize,
        execution_mode: SamplerExecutionMode,
    ) -> Self {
        let hook = EpisodeBoundHook::new(n_episodes);
        let env_builder = EnvBuilderType::homogeneous(builder, n_env);
        let sampler = DirectSampler::build(env_builder, hook, execution_mode);
        Self {
            sampler,
            _phantom: PhantomData,
        }
    }
}

impl<A: Actor + Clone> Evaluator<GymEnv, A> {
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

impl<E: Env, A: Actor + Clone> Evaluator<E, A> {
    /// Evaluates an actor and returns the collected trajectory views.
    #[allow(clippy::type_complexity)]
    pub fn eval(&mut self, actor: A) -> impl AsRef<[TrajectoryView<'_, E::Tensor>]> {
        let adapted_actor = ActorWrapper::new(actor);
        self.sampler.reset_all_envs();
        self.sampler.collect_rollouts(adapted_actor);
        self.sampler.trajectory_views()
    }
}
