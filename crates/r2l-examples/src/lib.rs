pub mod burn;
pub mod candle;

use std::{any::Any, f64};

const ENV_NAME: &str = "Pendulum-v1";

pub type EventBox = Box<dyn Any + Send + Sync>;

#[derive(Debug, Default, Clone)]
pub struct PPOProgress {
    pub clip_fractions: Vec<f32>,
    pub entropy_losses: Vec<f32>,
    pub policy_losses: Vec<f32>,
    pub value_losses: Vec<f32>,
    pub clip_range: f32,
    pub approx_kl: f32,
    pub explained_variance: f32,
    pub progress: f64,
    pub std: f32,
    pub avarage_reward: f32,
    pub learning_rate: f64,
}

impl PPOProgress {
    pub fn clear(&mut self) -> Self {
        std::mem::take(self)
    }
}
