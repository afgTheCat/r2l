use crate::builders::sampler_hooks2::EvaluatorNormalizerOptions;
use r2l_core::{
    env::Env,
    env_builder::{EnvBuilderTrait, EnvBuilderType},
    sampler::{
        CollectionType, R2lSampler,
        env_pools::{FixedSizeEnvPoolKind, VariableSizedEnvPoolKind},
    },
    tensor::R2lBuffer,
};

#[derive(Default)]
pub enum EnvPoolType {
    #[default]
    VecStep,
    VecVariable,
    ThreadStep,
    ThreadVariable,
}

// eigher this has to be build or it cannot have an assoc type
pub struct SamplerType {
    pub capacity: usize,
    pub hook_options: EvaluatorNormalizerOptions,
    pub env_pool_type: EnvPoolType,
}

impl SamplerType {
    // TODO: do we need the static bound here?
    pub fn build_with_builder_type<
        E: Env<Tensor = R2lBuffer> + 'static,
        EB: EnvBuilderTrait<Env = E>,
    >(
        &self,
        builder_type: EnvBuilderType<EB>,
    ) -> R2lSampler<EB::Env> {
        let n_envs = builder_type.num_envs();
        let collection_type = match self.env_pool_type {
            EnvPoolType::VecVariable => {
                let env_pool = builder_type.build_variable_sized_vec();
                let env_pool = VariableSizedEnvPoolKind::VariableSizedVecEnvPool(env_pool);
                CollectionType::EpisodeBound { env_pool }
            }
            EnvPoolType::VecStep => {
                let env_pool = builder_type.build_fixed_sized_vec(self.capacity);
                let env_desctiption = env_pool.env_description();
                let env_builder = builder_type.env_builder();
                let hooks = self
                    .hook_options
                    .build(env_desctiption, env_builder.as_ref(), n_envs);
                let env_pool = FixedSizeEnvPoolKind::FixedSizeVecEnvPool(env_pool);
                CollectionType::StepBound { env_pool, hooks }
            }
            EnvPoolType::ThreadVariable => {
                let env_pool = builder_type.build_variable_sized_thread();
                let env_pool = VariableSizedEnvPoolKind::VariableSizedThreadEnvPool(env_pool);
                CollectionType::EpisodeBound { env_pool }
            }
            EnvPoolType::ThreadStep => {
                let env_pool = builder_type.build_fixed_sized_thread(self.capacity);
                let env_desctiption = env_pool.env_description();
                let env_builder = builder_type.env_builder();
                let hooks = self
                    .hook_options
                    .build(env_desctiption, env_builder.as_ref(), n_envs);
                let env_pool = FixedSizeEnvPoolKind::FixedSizeThreadEnvPool(env_pool);
                CollectionType::StepBound { env_pool, hooks }
            }
        };
        R2lSampler {
            env_steps: self.capacity,
            collection_type,
        }
    }
}
