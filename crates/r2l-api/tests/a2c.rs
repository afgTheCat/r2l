use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use candle_core::Device;
use r2l_api::{
    builders::a2c::algorithm::A2CAlgorithmBuilder,
    hooks::{a2c::A2CStats, on_policy::LearningSchedule},
};
use r2l_gym::GymEnvBuilder;
use r2l_sampler::StepTrajectoryBound;

struct A2CTestConfig {
    env_name: &'static str,
    n_envs: usize,
    entropy_coeff: f32,
    gae_lambda: f32,
    gamma: f32,
    learning_rate: Option<f64>,
    gradient_clipping: Option<f32>,
    n_steps: usize,
    n_timesteps: usize,
    vf_coeff: Option<f32>,
    norm_obs: Option<bool>,
    norm_reward: Option<bool>,
    use_sde: Option<bool>,
    sde_sample_freq: Option<usize>,
}

fn configure_candle_ppo_test(config: A2CTestConfig) {
    let (update_tx, update_rx): (Sender<A2CStats>, Receiver<A2CStats>) = mpsc::channel();

    let mut a2c_builder = A2CAlgorithmBuilder::<GymEnvBuilder>::new(config.env_name, config.n_envs)
        .with_candle(Device::Cpu)
        .with_entropy_coeff(config.entropy_coeff)
        .with_lambda(config.gae_lambda)
        .with_gamma(config.gamma)
        .with_bound(StepTrajectoryBound::new(config.n_steps))
        .with_learning_schedule(LearningSchedule::total_step_bound(config.n_timesteps))
        .with_reporter(Some(update_tx));

    if let Some(learning_rate) = config.learning_rate {
        a2c_builder = a2c_builder.with_learning_rate(learning_rate);
    }

    if let Some(gradient_clipping) = config.gradient_clipping {
        a2c_builder = a2c_builder.with_gradient_clipping(Some(gradient_clipping));
    }

    if let Some(vf_coeff) = config.vf_coeff {
        a2c_builder = a2c_builder.with_vf_coeff(Some(vf_coeff));
    }

    let mut a2c = a2c_builder.build().unwrap();

    let t = thread::spawn(move || {
        while let Ok(stats) = update_rx.recv() {
            println!("avg reward: {}", stats.avarage_reward);
        }
    });

    a2c.train().unwrap();
    drop(a2c);
    t.join().unwrap();
}
