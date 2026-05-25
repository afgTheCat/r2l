use std::marker::PhantomData;

use r2l_core::env::{Env, EnvBuilder, EnvBuilderType};
use r2l_sampler::{
    SamplerExecutionMode,
    sampler2::{R2lSampler2, SamplerHook2},
};

use crate::hooks::sampler2::{EpisodeBoundHook, StepBoundHook};

/// Builder trait for sampler2 hook configurations.
///
/// Implementations of this trait package a rollout-collection policy into a
/// type that can later construct the concrete hook consumed by
/// [`R2lSampler2`]. This is the sampler2 equivalent of choosing a rollout
/// bound in the original sampler interface, but generalized to a hook-driven
/// collection model.
pub trait SamplerHook2Builder {
    /// Environment type collected by the resulting hook.
    type Env: Env;
    /// Concrete sampler hook produced by this builder.
    type Target: SamplerHook2<E = Self::Env>;

    /// Builds the hook used by [`Sampler2Builder`] when constructing a sampler.
    fn build(self) -> Self::Target;
}

/// Step-bounded sampler hook configuration.
///
/// This hook builder configures rollout collection to stop after a fixed
/// number of environment steps have been collected per active worker.
pub struct StepHookBound<E: Env> {
    n_step: usize,
    _phantom: PhantomData<E>,
}

impl<E: Env> StepHookBound<E> {
    /// Creates a step-bounded sampler hook configuration.
    pub fn new(n_step: usize) -> Self {
        Self {
            n_step,
            _phantom: PhantomData,
        }
    }
}

impl<E: Env> SamplerHook2Builder for StepHookBound<E> {
    type Env = E;
    type Target = StepBoundHook<Self::Env>;

    fn build(self) -> Self::Target {
        StepBoundHook::new(self.n_step)
    }
}

/// Episode-bounded sampler hook configuration.
///
/// This hook builder configures rollout collection to stop after a fixed
/// number of completed episodes have been collected per active worker.
pub struct EpisodeHookBound<E: Env> {
    n_episodes: usize,
    _phantom: PhantomData<E>,
}

impl<E: Env> EpisodeHookBound<E> {
    /// Creates an episode-bounded sampler hook configuration.
    pub fn new(n_episodes: usize) -> Self {
        Self {
            n_episodes,
            _phantom: PhantomData,
        }
    }
}

impl<E: Env> SamplerHook2Builder for EpisodeHookBound<E> {
    type Env = E;
    type Target = EpisodeBoundHook<Self::Env>;

    fn build(self) -> Self::Target {
        EpisodeBoundHook::new(self.n_episodes)
    }
}

/// Builder for [`R2lSampler2`] instances.
///
/// This builder configures how environments are instantiated, which
/// hook-driven rollout policy controls collection, and where sampler execution
/// takes place.
///
/// By default, [`new`](Self::new) creates a homogeneous vectorized sampler
/// using `n_envs` copies of the same environment builder, a
/// [`StepHookBound`] of `1024`, and [`SamplerExecutionMode::Vec`].
pub struct Sampler2Builder<EB: EnvBuilder, S: SamplerHook2Builder<Env = EB::Env>> {
    pub(crate) env_builder: EnvBuilderType<EB>,
    pub(crate) hook_builder: S,
    pub(crate) execution_mode: SamplerExecutionMode,
}

/// Default sampler2 builder using a step-bounded rollout policy.
pub type DefaultSamplerBuilder<EB> = Sampler2Builder<EB, StepHookBound<<EB as EnvBuilder>::Env>>;

impl<EB: EnvBuilder> DefaultSamplerBuilder<EB> {
    /// Creates a sampler builder from a single environment builder and count.
    ///
    /// The provided builder is replicated into a homogeneous environment set
    /// with `n_envs` copies.
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let env_builder = EnvBuilderType::homogenous(builder.into(), n_envs);
        Self {
            env_builder,
            hook_builder: StepHookBound::new(1024),
            execution_mode: SamplerExecutionMode::Vec,
        }
    }
}

impl<EB: EnvBuilder, S: SamplerHook2Builder<Env = EB::Env>> Sampler2Builder<EB, S> {
    /// Replaces the rollout hook policy used by the sampler.
    ///
    /// This changes the hook-builder type carried by the builder, allowing
    /// callers to swap between the standard step/episode hook bounds or install
    /// a custom sampler hook configuration.
    pub fn with_hook<S2: SamplerHook2Builder<Env = EB::Env>>(
        self,
        hook_builder: S2,
    ) -> Sampler2Builder<EB, S2> {
        let Sampler2Builder {
            env_builder,
            execution_mode,
            ..
        } = self;
        Sampler2Builder {
            env_builder,
            execution_mode,
            hook_builder,
        }
    }

    /// Sets where the sampler should execute.
    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.execution_mode = location;
        self
    }

    /// Replaces the full environment builder configuration.
    ///
    /// This is useful when you need heterogeneous environments or when the
    /// default homogeneous setup created by [`new`](DefaultSamplerBuilder::new)
    /// is not enough.
    pub fn with_env_builder(mut self, env_builder: EnvBuilderType<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

    /// Builds the configured sampler2 instance.
    pub fn build(self) -> R2lSampler2<EB::Env, S::Target> {
        let hook = self.hook_builder.build();
        R2lSampler2::build(self.env_builder, hook, self.execution_mode)
    }
}
