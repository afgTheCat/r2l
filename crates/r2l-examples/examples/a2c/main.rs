// ANCHOR: a2c
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use candle_core::Device;
use r2l::{A2CBuilder, A2CRolloutStats, SamplerExecutionMode, TrainingLimit};

fn main() -> anyhow::Result<()> {
    let (update_tx, update_rx): (Sender<A2CRolloutStats>, Receiver<A2CRolloutStats>) =
        mpsc::channel();

    let a2c_builder = A2CBuilder::gym("Pendulum-v1", 10)?
        .with_candle(Device::Cpu)
        .with_seed(0)
        .with_entropy_coefficient(0.2)
        .with_gradient_clipping(Some(0.5))
        .with_rollout_steps(2048)
        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
        .with_training_limit(TrainingLimit::rollouts(300))
        .with_rollout_reporter(Some(update_tx));
    let mut a2c = a2c_builder.build()?;
    let t = thread::spawn(move || {
        while let Ok(stats) = update_rx.recv() {
            println!("avg reward: {}", stats.average_reward);
        }
    });
    a2c.train()?;
    drop(a2c);
    t.join()
        .map_err(|_| anyhow::anyhow!("A2C reporter thread panicked"))?;
    Ok(())
}
// ANCHOR_END: a2c
