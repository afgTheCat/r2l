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
use r2l_sampler::{Location, StepTrajectoryBound};

fn main() {
    // ANCHOR: a2c
    let (update_tx, update_rx): (Sender<A2CStats>, Receiver<A2CStats>) = mpsc::channel();
    let ppo_builder = A2CAlgorithmBuilder::<GymEnvBuilder>::new("Pendulum-v1", 10)
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
    // ANCHOR_END: a2c
}
