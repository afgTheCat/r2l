use r2l_core::{
    error::{Error, Result},
    tensor::R2lTensor,
};

use crate::Network;

pub mod bernoulli;
mod burn;
mod candle;
pub mod categorical;
pub mod composite;
pub mod diagonal;
pub mod multi_categorical;
pub mod policy;

/// A tensor value held directly or in a backend's trainable parameter wrapper.
pub trait TensorParameter<T: R2lTensor>: Clone + std::fmt::Debug + Send + 'static {
    /// Returns the current tensor while preserving its gradient graph.
    fn value(&self) -> T;
}

impl<T: R2lTensor> TensorParameter<T> for T {
    fn value(&self) -> T {
        self.clone()
    }
}

fn output_width<N: Network>(network: &N) -> Result<usize> {
    let shape = network.output_shape();
    match shape.dims() {
        &[width] if width > 0 => Ok(width),
        _ => Err(Error::invalid_parameter(
            "distribution output shape",
            "[positive output width]",
            format!("{shape:?}"),
        )),
    }
}

fn single_observation<T: R2lTensor>(observation: &T) -> Result<()> {
    let shape = observation.to_shape();
    if !matches!(shape.dims(), [1, _]) {
        return Err(Error::invalid_parameter(
            "action input shape",
            "[1, features]",
            format!("{shape:?}"),
        ));
    }
    Ok(())
}

fn action_batch<T: R2lTensor>(actions: &T, batch: usize, width: usize) -> Result<()> {
    let shape = actions.to_shape();
    if shape.dims() != [batch, width] {
        return Err(Error::invalid_parameter(
            "action shape",
            format!("[{batch}, {width}]"),
            format!("{shape:?}"),
        ));
    }
    Ok(())
}
