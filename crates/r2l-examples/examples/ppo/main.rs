// ANCHOR: ppo
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use candle_core::Device;
use r2l_api::{
    EpisodeTrajectoryBound, Location, StepTrajectoryBound,
    builders::{
        agent::AgentBuilder,
        ppo::{agent::PPOAgentBuilder, algorithm::PPOAlgorithmBuilder},
        sampler::SamplerBuilder,
    },
    hooks::{on_policy::LearningSchedule, ppo::PPOStats},
};
use r2l_core::env::ActionSpaceType;
use r2l_gym::GymEnvBuilder;

fn main() {
    let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
    let sampler_builder = SamplerBuilder::<GymEnvBuilder>::new(gym_env_builder, 10)
        .with_location(Location::Vec)
        .with_location(Location::Thread)
        .with_bound(EpisodeTrajectoryBound::new(10))
        .with_bound(StepTrajectoryBound::new(1000));
    let sampler = sampler_builder.build();

    let ppo_agent = PPOAgentBuilder::new(10)
        .with_candle(Device::Cpu)
        .with_burn()
        .with_normalize_advantage(true)
        .with_entropy_coeff(0.)
        .with_vf_coeff(None)
        .with_gradient_clipping(None)
        .with_gamma(0.)
        .with_lambda(0.)
        .with_policy_hidden_layers(vec![32, 32])
        .with_learning_rate(3e-4)
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_epsilon(1e-5)
        .with_target_kl(Some(0.3))
        .with_clip_range(0.5)
        .with_weight_decay(1e-4)
        .build(10, 2, ActionSpaceType::Discrete);

    let (update_tx, update_rx): (Sender<PPOStats>, Receiver<PPOStats>) = mpsc::channel();
    let ppo_builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 10)
        .with_normalize_advantage(true)
        .with_candle(Device::Cpu)
        .with_burn()
        .with_entropy_coeff(0.2)
        .with_gradient_clipping(Some(0.5))
        .with_target_kl(Some(0.01))
        .with_bound(StepTrajectoryBound::new(2048))
        .with_location(Location::Vec)
        .with_clip_range(0.2)
        .with_learning_schedule(LearningSchedule::rollout_bound(300))
        .with_reporter(Some(update_tx));
    let mut ppo = ppo_builder.build().unwrap();
    let t = thread::spawn(move || {
        while let Ok(stats) = update_rx.recv() {
            println!("avg reward: {}", stats.average_reward);
        }
    });
    ppo.train().unwrap();
    drop(ppo);
    t.join().unwrap();
}
// ANCHOR_END: ppo
