// ANCHOR: a2c
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use candle_core::Device;
use r2l_api::{
    A2CAlgorithmBuilder, A2CStats, LearningSchedule, SamplerExecutionMode, StepTrajectoryBound,
};

fn main() {
    let (update_tx, update_rx): (Sender<A2CStats>, Receiver<A2CStats>) = mpsc::channel();

    let a2c_builder = A2CAlgorithmBuilder::gym("Pendulum-v1", 10)
        .with_candle(Device::Cpu)
        .with_burn()
        .with_entropy_coeff(0.2)
        .with_gradient_clipping(Some(0.5))
        .with_bound(StepTrajectoryBound::new(2048))
        .with_execution_mode(SamplerExecutionMode::Vec)
        .with_learning_schedule(LearningSchedule::rollout_bound(300))
        .with_reporter(Some(update_tx));
    let mut ppo = a2c_builder.build().unwrap();
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
