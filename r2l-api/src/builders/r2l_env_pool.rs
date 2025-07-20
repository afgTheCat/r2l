use crate::{
    builders::env_pool::{BuilderType, EnvBuilderTrait, VecPoolType},
    hooks::sequential_env_hooks::EmptySequentialVecEnv,
};
use candle_core::{Device, Result};
use r2l_core::env::{
    Env,
    orchestrator::{R2lEnvHolder, R2lEnvPool, StepMode, VecEnvHolder},
};

impl VecPoolType {
    fn to_r2l_pool<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
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
            _ => todo!(),
        }
    }
}
