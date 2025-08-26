use crate::{
    distributions::Distribution,
    env::{Env, Sampler},
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Result, Tensor};

pub struct VariableSizedTrajectoryBuff<E: Env> {
    states: Vec<E::Tensor>,
    next_states: Vec<E::Tensor>,
    rewards: Vec<f32>,
    actions: Vec<f32>,
    terminated: Vec<bool>,
    trancuated: Vec<bool>,
}

pub struct VariableSizedStateBufferWithEnv<E: Env> {
    env: E,
    buf: VariableSizedTrajectoryBuff<E>,
}

pub trait VariableSizedEnvPool {
    type Env: Env;
}

pub struct VariableSizedVecEnvPool<E: Env> {
    buffs: Vec<VariableSizedStateBufferWithEnv<E>>,
}

impl<E: Env> VariableSizedEnvPool for VariableSizedVecEnvPool<E> {
    type Env = E;
}

pub struct VariableSizedSampler<E: Env, P: VariableSizedEnvPool<Env = E>> {
    env_pool: P,
}

impl<E: Env, P: VariableSizedEnvPool<Env = E>> Sampler for VariableSizedSampler<E, P> {
    fn collect_rollouts<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distribution: &D,
    ) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }
}
