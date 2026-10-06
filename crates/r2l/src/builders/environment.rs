use std::num::NonZeroUsize;

use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType, EnvDescription, normalizer::Normalizer},
    error::Error,
};
use r2l_sampler::{DirectSamplerCore, SamplerExecutionMode, StagedSamplerCore};

use crate::evaluator::EvaluationSampler;

pub(crate) trait EnvBuildPlan<E: Env>: Send {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: NonZeroUsize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<EvaluationSampler<E>, Error>;

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<E>;

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<StagedSamplerCore<E>, Error>;

    fn env_description(&self) -> EnvDescription<E::Tensor>;

    fn n_envs(&self) -> NonZeroUsize;
}

pub(crate) struct TypedEnvBuildPlan<EB: EnvBuilder> {
    pub(crate) env_builder: EnvBuilderType<EB>,
}

impl<EB: EnvBuilder> EnvBuildPlan<EB::Env> for TypedEnvBuildPlan<EB> {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: NonZeroUsize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<<EB::Env as Env>::Tensor>>,
    ) -> Result<EvaluationSampler<EB::Env>, Error> {
        EvaluationSampler::build(
            self.env_builder.clone(),
            episodes_per_evaluation,
            evaluation_execution_mode,
            obs_normalizer,
        )
    }

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<EB::Env> {
        DirectSamplerCore::build(self.env_builder.clone(), execution_mode)
    }

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<<EB::Env as Env>::Tensor>>,
    ) -> Result<StagedSamplerCore<EB::Env>, Error> {
        StagedSamplerCore::build(&self.env_builder, execution_mode, obs_normalizer)
    }

    fn env_description(&self) -> EnvDescription<<EB::Env as Env>::Tensor> {
        self.env_builder.env_description().unwrap()
    }

    fn n_envs(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.env_builder.num_envs()).unwrap()
    }
}
