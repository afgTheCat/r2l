use candle_core::Device;
use r2l_agents::candle_agents::ppo3::PPO3DefaultHooks;
use r2l_api::builders::agents::ppo::PPOBuilder;
use r2l_core::{
    Algorithm,
    on_policy_algorithm::{
        DefaultOnPolicyAlgorightmsHooks, DefaultOnPolicyAlgorightmsHooks3, LearningSchedule,
        OnPolicyAlgorithm, OnPolicyAlgorithm3,
    },
    sampler::{CollectionType, R2lSampler, env_pools::FixedSizeEnvPoolKind},
    sampler2::{CollectionBound, env_pools::builder::EnvBuilderType2},
    sampler3::{R2lSamplerX, coordinator::Location},
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use std::{fs::File, sync::Arc};

#[test]
fn old_api() {
    let builder = EnvBuilderType2::EnvBuilder {
        builder: Arc::new(GymEnvBuilder::new("CartPole-v1")),
        n_envs: 10,
    };
    let env_pool = builder.build_fixed_sized_vec(2048);
    let env_description = env_pool.env_description();
    let env_pool = FixedSizeEnvPoolKind::FixedSizeVecEnvPool(env_pool);
    let sampler = R2lSampler {
        env_steps: 2048,
        collection_type: CollectionType::StepBound {
            env_pool,
            hooks: None,
        },
    };
    let agent = PPOBuilder::default()
        .build(&Device::Cpu, &env_description)
        .unwrap();
    let hooks = DefaultOnPolicyAlgorightmsHooks::new(LearningSchedule::RolloutBound {
        total_rollouts: 10,
        current_rollout: 0,
    });
    let mut op = OnPolicyAlgorithm {
        sampler,
        agent,
        hooks,
    };
    op.train().unwrap()
}

fn run_test() {
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
        total_rollouts: 10,
        current_rollout: 0,
    });
    let mut op3 = OnPolicyAlgorithm3 {
        sampler,
        agent,
        hooks,
    };
    op3.train().unwrap();
}

#[test]
fn new_new_api() {
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();

    run_test();

    // if let Ok(report) = guard.report().build() {
    //     let file = File::create("flamegraph.svg").unwrap();
    //     report.flamegraph(file).unwrap();
    // };
}
