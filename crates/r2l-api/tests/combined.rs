use std::time::{Duration, Instant};

use r2l_api::{
    LearningSchedule, LearningSchedule2, PPO2AlgorithmBuilder, PPOAlgorithmBuilder, StepHookBound,
    StepTrajectoryBound,
};

#[allow(dead_code)]
#[derive(Clone, Copy)]
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
    // TODO: implement these features
    norm_obs: Option<bool>,
    norm_reward: Option<bool>,
    use_sde: Option<bool>,
    sde_sample_freq: Option<usize>,
}

fn new_configure_candle_ppo_test(config: PPOTestConfig) {
    let mut ppo_builder = PPO2AlgorithmBuilder::gym(config.env_name, config.n_envs)
        .with_candle(candle_core::Device::Cpu)
        .with_entropy_coeff(config.entropy_coeff)
        .with_lambda(config.gae_lambda)
        .with_gamma(config.gamma)
        .with_total_epochs(config.total_epochs)
        .with_rollout_bound(StepHookBound::new(config.n_steps))
        .with_learning_schedule(LearningSchedule2::total_step_bound(config.n_timesteps));

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

    let mut ppo = ppo_builder.build().unwrap();
    ppo.train().unwrap();
}

fn old_configure_candle_ppo_test(config: PPOTestConfig) {
    let mut ppo_builder = PPOAlgorithmBuilder::gym(config.env_name, config.n_envs)
        .with_candle(candle_core::Device::Cpu)
        .with_entropy_coeff(config.entropy_coeff)
        .with_lambda(config.gae_lambda)
        .with_gamma(config.gamma)
        .with_total_epochs(config.total_epochs)
        .with_rollout_bound(StepTrajectoryBound::new(config.n_steps))
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

    let mut ppo = ppo_builder.build().unwrap();
    ppo.train().unwrap();
}

fn average_duration(total: Duration, runs: u32) -> Duration {
    Duration::from_secs_f64(total.as_secs_f64() / runs as f64)
}

#[test]
fn cartpole_runtime_compare() {
    // Source: Stable-Baselines3 / RL Zoo reference captured in envs_to_test.txt
    // https://huggingface.co/sb3/ppo-CartPole-v1
    let test_config = PPOTestConfig {
        env_name: "CartPole-v1",
        n_envs: 8,
        clip_range: Some(0.2),
        entropy_coeff: 0.0,
        gae_lambda: 0.8,
        gamma: 0.98,
        learning_rate: Some(0.001),
        sample_size: Some(256),
        total_epochs: 20,
        n_steps: 32,
        n_timesteps: 100000,
        vf_coeff: None,
        gradient_clipping: None,
        norm_obs: Some(false),
        norm_reward: None,
        use_sde: None,
        sde_sample_freq: None,
    };

    const NUM_RUNS: u32 = 10;
    let mut old_total = Duration::ZERO;
    let mut new_total = Duration::ZERO;

    for _ in 0..NUM_RUNS {
        let start = Instant::now();
        old_configure_candle_ppo_test(test_config);
        old_total += start.elapsed();
    }

    for _ in 0..NUM_RUNS {
        let start = Instant::now();
        new_configure_candle_ppo_test(test_config);
        new_total += start.elapsed();
    }

    let old_avg = average_duration(old_total, NUM_RUNS);
    let new_avg = average_duration(new_total, NUM_RUNS);
    let ratio = new_total.as_secs_f64() / old_total.as_secs_f64();

    println!(
        "cartpole runtime compare over {NUM_RUNS} runs: old_total={old_total:?}, new_total={new_total:?}, old_avg={old_avg:?}, new_avg={new_avg:?}, new/old={ratio:.3}x"
    );
}

#[test]
fn pendulum_runtime_compare() {
    // Source: Stable-Baselines3 / RL Zoo reference captured in envs_to_test.txt
    // https://huggingface.co/sb3/ppo-Pendulum-v1
    let test_config = PPOTestConfig {
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
        norm_obs: Some(false),
        norm_reward: None,
        use_sde: Some(true),
        sde_sample_freq: Some(4),
    };

    const NUM_RUNS: u32 = 10;
    let mut old_total = Duration::ZERO;
    let mut new_total = Duration::ZERO;

    for _ in 0..NUM_RUNS {
        let start = Instant::now();
        old_configure_candle_ppo_test(test_config);
        old_total += start.elapsed();
    }

    for _ in 0..NUM_RUNS {
        let start = Instant::now();
        new_configure_candle_ppo_test(test_config);
        new_total += start.elapsed();
    }

    let old_avg = average_duration(old_total, NUM_RUNS);
    let new_avg = average_duration(new_total, NUM_RUNS);
    let ratio = new_total.as_secs_f64() / old_total.as_secs_f64();

    println!(
        "pendulum runtime compare over {NUM_RUNS} runs: old_total={old_total:?}, new_total={new_total:?}, old_avg={old_avg:?}, new_avg={new_avg:?}, new/old={ratio:.3}x"
    );
}
