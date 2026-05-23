use std::marker::PhantomData;

use r2l_core::env::{Env, EnvBuilder, EnvBuilderType};
use r2l_sampler::{
    SamplerExecutionMode,
    sampler2::{R2lSampler2, SamplerHook2},
};

use crate::hooks::sampler2::{EpisodeBoundHook, StepBoundHook};

pub trait SamplerHook2Builder {
    type Env: Env;
    type Target: SamplerHook2<E = Self::Env>;

    fn build(self) -> Self::Target;
}

pub struct StepHookBound<E: Env> {
    n_step: usize,
    _phantom: PhantomData<E>,
}

impl<E: Env> StepHookBound<E> {
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

pub struct EpisodeHookBound<E: Env> {
    n_episodes: usize,
    _phantom: PhantomData<E>,
}

impl<E: Env> EpisodeHookBound<E> {
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

pub struct Sampler2Builder<EB: EnvBuilder, S: SamplerHook2Builder<Env = EB::Env>> {
    pub(crate) env_builder: EnvBuilderType<EB>,
    pub(crate) hook_builder: S,
    pub(crate) execution_mode: SamplerExecutionMode,
}

pub type DefaultSamplerBuilder<EB> = Sampler2Builder<EB, StepHookBound<<EB as EnvBuilder>::Env>>;

impl<EB: EnvBuilder> DefaultSamplerBuilder<EB> {
    pub(crate) fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let env_builder = EnvBuilderType::homogenous(builder.into(), n_envs);
        Self {
            env_builder,
            hook_builder: StepHookBound::new(1024),
            execution_mode: SamplerExecutionMode::Vec,
        }
    }
}

impl<EB: EnvBuilder, S: SamplerHook2Builder<Env = EB::Env>> Sampler2Builder<EB, S> {
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

    pub fn with_execution_mode(mut self, location: SamplerExecutionMode) -> Self {
        self.execution_mode = location;
        self
    }

    pub fn with_env_builder(mut self, env_builder: EnvBuilderType<EB>) -> Self {
        self.env_builder = env_builder;
        self
    }

    pub fn build(self) -> R2lSampler2<EB::Env, S::Target> {
        let hook = self.hook_builder.build();
        R2lSampler2::build(self.env_builder, hook, self.execution_mode)
    }
}
