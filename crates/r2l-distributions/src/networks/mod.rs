use std::fmt::Debug;

use r2l_core::{
    Shape,
    error::{Error, Result},
    tensor::R2lTensor,
};

pub mod cnn;
pub mod mlp;

/// A network with input and output shapes defined per observation, without a batch axis.
///
/// Inputs use flattened rows `[batch, input_shape.num_elements()]`, including CNN inputs.
/// A successful forward pass must preserve the nonzero batch size and return
/// `[batch, ..output_shape]`. Shapes must remain stable for the lifetime of the network.
pub trait Network: Send + Debug + Clone + 'static {
    type Tensor: R2lTensor;

    fn input_shape(&self) -> Shape;
    fn output_shape(&self) -> Shape;

    /// Evaluates a nonempty batch, preserving gradients through network parameters.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible input dimensions or failed evaluation.
    fn forward(&self, t: Self::Tensor) -> Result<Self::Tensor>;

    /// Validates flattened input rows and returns the batch size.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor with a batch axis followed by flattened observation features.
    ///
    /// # Errors
    ///
    /// Returns an error for empty batches or incompatible observation dimensions.
    fn batch_size(&self, input: &Self::Tensor) -> Result<usize> {
        let shape = input.to_shape();
        let size = self.input_shape().num_elements();
        match shape.dims() {
            &[batch, features] if batch > 0 && features > 0 && features == size => Ok(batch),
            _ => Err(Error::invalid_parameter(
                "network input shape",
                format!("[nonzero batch, flattened {:?}]", self.input_shape()),
                format!("{shape:?}"),
            )),
        }
    }
}
