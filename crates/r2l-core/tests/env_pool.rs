use r2l_core::{
    env::Env,
    sampler2::{Buffer, EmptyPreProcessor, R2lSampler2, env_pools::ThreadEnvPool},
};

type SamplerWithThreads<E, B> = R2lSampler2<ThreadEnvPool<E, B>, EmptyPreProcessor>;

fn thing<E: Env, B: Buffer<E = E> + Send + Clone>(t: SamplerWithThreads<E, B>) {}

#[test]
fn do_the_thing() {}
