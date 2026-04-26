// ANCHOR: ppo
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use candle_core::Device;
use r2l_api::{
    LearningSchedule, PPOAlgorithmBuilder, PPOStats, SamplerExecutionMode, StepTrajectoryBound,
};

fn main() {
    let (update_tx, update_rx): (Sender<PPOStats>, Receiver<PPOStats>) = mpsc::channel();
    let ppo_builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 10)
        .with_normalize_advantage(true)
        .with_candle(Device::Cpu)
        .with_burn()
        .with_entropy_coeff(0.2)
        .with_gradient_clipping(Some(0.5))
        .with_target_kl(Some(0.01))
        .with_bound(StepTrajectoryBound::new(2048))
        .with_execution_mode(SamplerExecutionMode::Vec)
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
