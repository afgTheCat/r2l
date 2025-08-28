use candle_core::Device;
use candle_core::Result;
use r2l_core::{env::Env, numeric::Buffer};
use r2l_gym::GymEnv;

pub trait EnvBuilderTrait: Sync + Send + 'static {
    type Env: Env<Tensor = Buffer>;

    fn build_env(&self, device: &Device) -> Result<Self::Env>;
}

impl EnvBuilderTrait for String {
    type Env = GymEnv;

    fn build_env(&self, _device: &Device) -> Result<Self::Env> {
        Ok(GymEnv::new(&self, None))
    }
}

impl<E: Env<Tensor = Buffer>, F: Sync + Send + 'static> EnvBuilderTrait for F
where
    F: Fn(&Device) -> Result<E>,
{
    type Env = E;

    fn build_env(&self, device: &Device) -> Result<Self::Env> {
        (self)(device)
    }
}
