use r2l_core::{
    env::Env,
    sampler2::{Buffer, EmptyPreProcessor, R2lSampler2, env_pools::WorkerPool},
};

type Thing<E, B> = R2lSampler2<WorkerPool<E, B>, EmptyPreProcessor>;

fn thing<E: Env, B: Buffer<E = E> + Send + Clone>(t: Thing<E, B>) {}

#[test]
fn do_the_thing() {}
