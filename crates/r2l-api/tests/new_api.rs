use std::sync::Arc;

use r2l_core::sampler2::{
    CollectionBound, R2lSampler2,
    env_pools::builder::{BufferType, EnvBuilderType2, EnvPoolBuilder, WorkerLocation},
};
use r2l_gym::GymEnvBuilder;

#[test]
fn new_api() {
    let builder = EnvBuilderType2::EnvBuilder {
        builder: Arc::new(GymEnvBuilder::new("CartPole-v1")),
        n_envs: 10,
    };
    // now this is something
    let env_pool_builder = EnvPoolBuilder {
        worker_location: WorkerLocation::Thread,
        collection_bound: CollectionBound::StepBound { steps: 2048 },
        buffer_type: BufferType::FixedSize,
    };

    let env_pool = env_pool_builder.build(builder);
    let sampler = R2lSampler2::new(env_pool, None);
}
