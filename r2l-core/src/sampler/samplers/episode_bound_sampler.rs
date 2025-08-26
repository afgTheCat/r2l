use crate::{
    env::{Env, Sampler},
    numeric::Buffer,
    sampler::env_pools::VariableSizedEnvPool,
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Result, Tensor};

pub struct EpisodeBoundAsyncSampler<E: Env<Tensor = Buffer>, P: VariableSizedEnvPool<Env = E>> {
    pub required_steps: usize,
    pub env_pool: P,
}

impl<E: Env<Tensor = Buffer>, P: VariableSizedEnvPool<Env = E>> Sampler
    for EpisodeBoundAsyncSampler<E, P>
{
    fn collect_rollouts<D: crate::distributions::Distribution<Tensor = Tensor>>(
        &mut self,
        distribution: &D,
    ) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }
}

pub struct EpisodeBoundSequentialSampler<E: Env<Tensor = Buffer>, P: VariableSizedEnvPool<Env = E>>
{
    pub required_steps: usize,
    pub env_pool: P,
}

impl<E: Env<Tensor = Buffer>, P: VariableSizedEnvPool<Env = E>> Sampler
    for EpisodeBoundSequentialSampler<E, P>
{
    fn collect_rollouts<D: crate::distributions::Distribution<Tensor = Tensor>>(
        &mut self,
        distribution: &D,
    ) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }
}
