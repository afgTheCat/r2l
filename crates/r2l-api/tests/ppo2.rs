use r2l_api::{LearningSchedule, PPOAlgorithmBuilder, StepHookBound};
use r2l_core::{
    env::EnvBuilderType,
    on_policy::algorithm::{DefaultAdapter, OnPolicyAlgorithm, OnPolicyRuntime},
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{R2lNormalizedSampler, SamplerExecutionMode};

#[allow(dead_code)]
struct PPOTestConfig {
    env_name: &'static str,
    n_envs: usize,
    clip_range: Option<f32>,
    entropy_coeff: f32,
    gae_lambda: f32,
    gamma: f32,
    learning_rate: Option<f64>,
    sample_size: Option<usize>,
    total_epochs: usize,
    n_steps: usize,
    n_timesteps: usize,
    vf_coeff: Option<f32>,
    gradient_clipping: Option<f32>,
}

fn configure_candle_ppo_test(config: PPOTestConfig) {
    let mut ppo_builder = PPOAlgorithmBuilder::gym(config.env_name, config.n_envs)
        .with_candle(candle_core::Device::Cpu)
        .with_entropy_coeff(config.entropy_coeff)
        .with_lambda(config.gae_lambda)
        .with_gamma(config.gamma)
        .with_total_epochs(config.total_epochs)
        .with_rollout_bound(StepHookBound::new(config.n_steps))
        .with_learning_schedule(LearningSchedule::total_step_bound(config.n_timesteps));

    if let Some(clip_range) = config.clip_range {
        ppo_builder = ppo_builder.with_clip_range(clip_range);
    }

    if let Some(learning_rate) = config.learning_rate {
        ppo_builder = ppo_builder.with_learning_rate(learning_rate);
    }

    if let Some(sample_size) = config.sample_size {
        ppo_builder = ppo_builder.with_sample_size(sample_size);
    }

    if let Some(vf_coeff) = config.vf_coeff {
        ppo_builder = ppo_builder.with_vf_coeff(Some(vf_coeff));
    }

    if let Some(gradient_clipping) = config.gradient_clipping {
        ppo_builder = ppo_builder.with_gradient_clipping(Some(gradient_clipping));
    }

    let ppo = ppo_builder.build().unwrap();
    let sampler = R2lNormalizedSampler::build(
        EnvBuilderType::homogenous(GymEnvBuilder::new(config.env_name), config.n_envs),
        config.n_steps,
        SamplerExecutionMode::Vec,
        false,
        false,
    );
    let runtime = OnPolicyRuntime {
        agent: ppo.runtime.agent,
        sampler,
        adapter: DefaultAdapter,
    };
    let hooks = r2l_api::DefaultOnPolicyAlgorithmHooks::new::<GymEnvBuilder>(
        LearningSchedule::total_step_bound(config.n_timesteps),
        None,
    );
    let mut ppo: OnPolicyAlgorithm<
        _,
        R2lNormalizedSampler<GymEnv>,
        r2l_api::DefaultOnPolicyAlgorithmHooks<_, R2lNormalizedSampler<GymEnv>, _, GymEnv>,
    > = OnPolicyAlgorithm { runtime, hooks };

    ppo.train().unwrap();
}

#[test]
fn mountain_car_continuous_candle_normalized_sampler() {
    configure_candle_ppo_test(PPOTestConfig {
        env_name: "MountainCarContinuous-v0",
        n_envs: 1,
        clip_range: Some(0.1),
        entropy_coeff: 0.00429,
        gae_lambda: 0.9,
        gamma: 0.9999,
        learning_rate: Some(7.77e-05),
        sample_size: Some(256),
        total_epochs: 10,
        n_steps: 8,
        n_timesteps: 20000,
        vf_coeff: Some(0.19),
        gradient_clipping: Some(5.0),
    });
}

#[test]
fn pendulum_candle_normalized_sampler() {
    configure_candle_ppo_test(PPOTestConfig {
        env_name: "Pendulum-v1",
        n_envs: 4,
        clip_range: Some(0.2),
        entropy_coeff: 0.0,
        gae_lambda: 0.95,
        gamma: 0.9,
        learning_rate: Some(0.001),
        sample_size: None,
        total_epochs: 10,
        n_steps: 1024,
        n_timesteps: 100000,
        vf_coeff: None,
        gradient_clipping: None,
    });
}
