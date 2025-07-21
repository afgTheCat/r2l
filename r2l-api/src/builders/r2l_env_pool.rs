use crate::{
    builders::env_pool::{BuilderType, EnvBuilderTrait, SequentialEnvHookTypes, VecPoolType},
    hooks::env_pool::EmptySequentialVecEnv,
};
use candle_core::{Device, Result};
use r2l_core::env::{
    Env,
    orchestrator::{R2lEnvHolder, R2lEnvPool, StepMode, VecEnvHolder},
    sequential_vec_env::SequentialVecEnvHooks,
};

impl VecPoolType {
    pub fn to_r2l_pool_inner<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        &self,
        device: &Device,
        env_builder: BuilderType<EB>,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        match self {
            Self::Dummy => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers()?;
                let env_description = envs[0].env_description();
                Ok(R2lEnvPool {
                    env_holder: R2lEnvHolder::Vec(VecEnvHolder { envs, buffers }),
                    step_mode: StepMode::Async,
                    env_description,
                })
            }
            Self::Sequential(hook_types) => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers()?;
                let env_description = envs[0].env_description();
                let hooks: Box<dyn SequentialVecEnvHooks> = match hook_types {
                    SequentialEnvHookTypes::None => Box::new(EmptySequentialVecEnv),
                    // TODO: this should also be a hook
                    // SequentialEnvHookTypes::NormalizerOnly { options } => {
                    //     let normalizer = options.build(env_description, self.n_envs, device);
                    //     Box::new(normalizer)
                    // }
                    SequentialEnvHookTypes::EvaluatorOnly { options } => {
                        let eval_env = env_builder.build_single_env()?;
                        let n_envs = env_builder.n_envs();
                        let evaluator = options.build(eval_env, n_envs);
                        Box::new(evaluator)
                    }
                    SequentialEnvHookTypes::EvaluatorNormalizer { options } => {
                        let eval_env = env_builder.build_single_env()?;
                        let n_envs = env_builder.n_envs();
                        let eval_normalizer = options.build(eval_env, n_envs, device.clone());
                        Box::new(eval_normalizer)
                    }
                    _ => todo!(),
                };
                Ok(R2lEnvPool {
                    env_holder: R2lEnvHolder::Vec(VecEnvHolder { envs, buffers }),
                    step_mode: StepMode::Sequential(hooks),
                    env_description,
                })
                // todo!()
            }
            _ => todo!(),
        }
    }

    pub fn to_r2l_pool<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builder: EB,
        n_envs: usize,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        self.to_r2l_pool_inner(device, BuilderType::env_builder(env_builder, n_envs))
    }
}
