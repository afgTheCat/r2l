// ANCHOR: a2c
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use candle_core::Device;
use r2l_api::{
    Location, StepTrajectoryBound,
    builders::{
        a2c::{agent::A2CAgentBuilder, algorithm::A2CAlgorithmBuilder},
        agent::AgentBuilder,
    },
    hooks::{a2c::A2CStats, on_policy::LearningSchedule},
};
use r2l_core::env::{ActionSpaceType, Space};

fn main() {
    let (update_tx, update_rx): (Sender<A2CStats>, Receiver<A2CStats>) = mpsc::channel();
    let a2c_algo = A2CAgentBuilder::new(10)
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
        .with_weight_decay(1e-4)
        .build(10, 2, ActionSpaceType::Discrete);

    let ppo_builder = A2CAlgorithmBuilder::gym("Pendulum-v1", 10)
        .with_candle(Device::Cpu)
        .with_burn()
        .with_entropy_coeff(0.2)
        .with_gradient_clipping(Some(0.5))
        .with_bound(StepTrajectoryBound::new(2048))
        .with_location(Location::Vec)
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
// ANCHOR_END: a2c
