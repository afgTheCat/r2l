// ANCHOR: a2c
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use candle_core::Device;
use r2l::{A2CAlgorithmBuilder, A2CRolloutStats, LearningSchedule, SamplerExecutionMode};

fn main() {
    let (update_tx, update_rx): (Sender<A2CRolloutStats>, Receiver<A2CRolloutStats>) =
        mpsc::channel();

    let a2c_builder = A2CAlgorithmBuilder::gym("Pendulum-v1", 10)
        .unwrap()
        .with_candle(Device::Cpu)
        .with_seed(0)
        .with_entropy_coeff(0.2)
        .with_gradient_clipping(Some(0.5))
        .with_rollout_steps(2048)
        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
        .with_learning_schedule(LearningSchedule::rollout_bound(300))
        .with_reporter(Some(update_tx));
    let mut a2c = a2c_builder.build().unwrap();
    let t = thread::spawn(move || {
        while let Ok(stats) = update_rx.recv() {
            println!("avg reward: {}", stats.average_reward);
        }
    });
    a2c.train().unwrap();
    drop(a2c);
    t.join().unwrap();
}
// ANCHOR_END: a2c
