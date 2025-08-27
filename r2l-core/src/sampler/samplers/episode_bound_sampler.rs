use crate::{
    distributions::Distribution,
    env::{Env, Sampler},
    numeric::Buffer,
    sampler::env_pools::{VariableSizedEnvPool, VariableSizedEnvPoolKind},
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Result, Tensor};

// TODO: for the episode bound sampler, we don't yet support hooks (and we might not want to). The
// only rollout mode that we support is gonna be the async one as well, since ther is just no
// reason for the environments to wait for each other.
pub struct EpisodeBoundSampler<E: Env<Tensor = Buffer>, P: VariableSizedEnvPool<Env = E>> {
    pub required_steps: usize,
    pub env_pool: P,
}

impl<E: Env<Tensor = Buffer>, P: VariableSizedEnvPool<Env = E>> Sampler
    for EpisodeBoundSampler<E, P>
{
    fn collect_rollouts<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distribution: &D,
    ) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }
}

pub type EpisodeBoundSamplerKind<E: Env<Tensor = Buffer>> =
    EpisodeBoundSampler<E, VariableSizedEnvPoolKind<E>>;
