use candle_core::Device;
use r2l_agents::candle_agents::ppo3::PPO3DefaultHooks;
use r2l_api::builders::agents::ppo::PPOBuilder;
use r2l_core::{
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks3, LearningSchedule, OnPolicyAlgorithm3},
    sampler2::{
        CollectionBound, R2lSampler2,
        env_pools::builder::{BufferType, EnvBuilderType2, EnvPoolBuilder, WorkerLocation},
    },
    sampler3::{R2lSamplerX, coordinator::Location},
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use std::sync::Arc;

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

#[test]
fn new_new_api() {
    let builder = EnvBuilderType2::EnvBuilder {
        builder: Arc::new(GymEnvBuilder::new("CartPole-v1")),
        n_envs: 10,
    };

    let sampler: R2lSamplerX<GymEnv> = R2lSamplerX::build(
        builder,
        CollectionBound::StepBound { steps: 2048 },
        None,
        Location::Vec,
    );
    let env_description = sampler.env_description();
    let agent = PPOBuilder::default()
        .build3(&Device::Cpu, &env_description, PPO3DefaultHooks::new())
        .unwrap();
    let hooks = DefaultOnPolicyAlgorightmsHooks3::new(LearningSchedule::RolloutBound {
        total_rollouts: 100,
        current_rollout: 0,
    });
    let mut op3 = OnPolicyAlgorithm3 {
        sampler,
        agent,
        hooks,
    };
    op3.train().unwrap();
}
