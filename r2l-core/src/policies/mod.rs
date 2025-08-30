use candle_core::{Result, Tensor};

pub trait ValueFunction {
    fn calculate_values(&self, observation: &Tensor) -> Result<Tensor>;
}
