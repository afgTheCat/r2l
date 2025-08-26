use crate::{
    distributions::Distribution,
    env::Env,
    numeric::Buffer,
    sampler::trajectory_buffers::{
        fixed_size_buffer::FixedSizeStateBuffer, variable_size_buffer::VariableSizedStateBuffer,
    },
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;

pub mod thread_env_pool;
pub mod vec_env_pool;

pub trait FixedSizeEnvPool {
    type Env: Env<Tensor = Buffer>;

    fn num_envs(&self) -> usize;

    // Single steps all the environments and returns the underlying state buffers
    fn single_step<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>>;

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>);

    // TODO: should be removed once we have a trait for the trajectory buffers
    fn to_rollout_buffers(&mut self, steps_per_environment: usize) -> Vec<RolloutBuffer>;

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize);
}

pub trait VariableSizedEnvPool {
    type Env: Env<Tensor = Buffer>;

    fn num_envs(&self) -> usize;

    fn single_step<D: Distribution<Tensor = Tensor>>(
        &mut self,
        env_idx: &[usize],
        distr: &D,
    ) -> Vec<VariableSizedStateBuffer<Self::Env>>;

    // TODO: should be removed once we have a trait for the trajectory buffers
    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer>;
}
