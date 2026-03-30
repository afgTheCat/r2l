pub mod burn;
pub mod candle;

use std::{any::Any, f64};

use r2l_api::hooks::ppo::BatchStats;

const ENV_NAME: &str = "Pendulum-v1";

pub type EventBox = Box<dyn Any + Send + Sync>;

#[derive(Debug, Default, Clone)]
pub struct PPOStatsOld {
    pub batch_stats: Vec<BatchStats>,
    pub std: f32,
    pub avarage_reward: f32,
    pub learning_rate: f64,
}

impl PPOStatsOld {
    pub fn clear(&mut self) -> Self {
        std::mem::take(self)
    }
}
