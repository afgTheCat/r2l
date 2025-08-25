use std::marker::PhantomData;

use crate::{distributions::Distribution, env::Env, numeric::Buffer};
use candle_core::Tensor;

// TODO: we need to extend this in the future
pub trait VariableSizedEnvPool {
    type Env: Env<Tensor = Buffer>;

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize);
}

pub struct VariableSizedVecEnvPool<E: Env> {
    // TODO: we can figure this out once later on
    buffers: PhantomData<E>,
}

impl<E: Env<Tensor = Buffer>> VariableSizedEnvPool for VariableSizedVecEnvPool<E> {
    type Env = E;

    fn run_rollouts<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        todo!()
    }
}
