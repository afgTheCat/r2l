use crate::{distributions::Policy, env::Env, sampler3::buffers::Buffer};

pub trait Preprocessor<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> {
    fn preprocess_states(&mut self, policy: &dyn Policy<Tensor = E::Tensor>, buffers: &mut Vec<B>);
}
