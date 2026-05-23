use r2l_core::env::{EnvBuilder, EnvBuilderType};
use r2l_sampler::{
    SamplerExecutionMode,
    sampler2::{R2lSampler2, SamplerHook2},
};

use crate::hooks::sampler2::StepBoundHook;

pub struct Sampler2Builder<EB: EnvBuilder, S: SamplerHook2<E = EB::Env>> {
    pub(crate) env_builder: EnvBuilderType<EB>,
    pub(crate) hook: S,
    pub(crate) execution_mode: SamplerExecutionMode,
}

pub type DefaultSamplerBuilder<EB: EnvBuilder> = Sampler2Builder<EB, StepBoundHook<EB::Env>>;

impl<EB: EnvBuilder> DefaultSamplerBuilder<EB> {
    pub(crate) fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let env_builder = EnvBuilderType::homogenous(builder.into(), n_envs);
        Self {
            env_builder,
            hook: StepBoundHook::new(1024),
            execution_mode: SamplerExecutionMode::Vec,
        }
    }
}

impl<EB: EnvBuilder, S: SamplerHook2<E = EB::Env>> Sampler2Builder<EB, S> {
    pub fn with_hook<S2: SamplerHook2<E = EB::Env>>(mut self, hook: S2) -> Sampler2Builder<EB, S2> {
        let Sampler2Builder {
            env_builder,
            execution_mode,
            ..
        } = self;
        Sampler2Builder {
            env_builder,
            execution_mode,
            hook,
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

    pub fn build(self) -> R2lSampler2<EB::Env, S> {
        R2lSampler2::build(self.env_builder, self.hook, self.execution_mode)
    }
}
