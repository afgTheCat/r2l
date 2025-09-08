use candle_core::Result;
use r2l_core::env::Env;
use r2l_gym::GymEnv;

pub trait EnvBuilderTrait: Sync + Send + 'static {
    type Env: Env;

    fn build_env(&self) -> Result<Self::Env>;
}

impl EnvBuilderTrait for String {
    type Env = GymEnv;

    fn build_env(&self) -> Result<Self::Env> {
        Ok(GymEnv::new(&self, None))
    }
}

impl<E: Env, F: Sync + Send + 'static> EnvBuilderTrait for F
where
    F: Fn() -> Result<E>,
{
    type Env = E;

    fn build_env(&self) -> Result<Self::Env> {
        (self)()
    }
}
