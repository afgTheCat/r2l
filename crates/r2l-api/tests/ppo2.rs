use r2l_api::{LearningSchedule2, PPO2AlgorithmBuilder, StepHookBound};

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
    // TODO: implement these features
    norm_obs: Option<bool>,
    norm_reward: Option<bool>,
    use_sde: Option<bool>,
    sde_sample_freq: Option<usize>,
}

fn configure_candle_ppo_test(config: PPOTestConfig) {
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

#[test]
fn new_cartpole_candle() {
    // Source: Stable-Baselines3 / RL Zoo reference captured in envs_to_test.txt
    // https://huggingface.co/sb3/ppo-CartPole-v1
    configure_candle_ppo_test(PPOTestConfig {
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
    });
}
