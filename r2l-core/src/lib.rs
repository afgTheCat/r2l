pub mod agents;
pub mod distributions;
pub mod env;
pub mod network;
pub mod on_policy_algorithm;
pub mod policies;
pub mod tensors;
pub mod thread_safe_sequential;
pub mod utils;

use candle_core::Result;

pub trait Algorithm {
    fn train(&mut self) -> Result<()>;
}
