use std::fmt::Debug;

use r2l_core::{Shape, error::Result, tensor::R2lTensor};

pub mod cnn;
pub mod mlp;

/// A network mapping flat observations to flat outputs.
///
/// Input and output shapes describe one batch element: `[input_size]` and
/// `[output_size]`, both with positive widths. Forwarding accepts
/// `[batch, input_size]` and returns `[batch, output_size]`, preserving the
/// nonzero batch size. CNNs reshape flat inputs internally using their spatial shape.
/// Declared shapes must remain stable for the lifetime of the network.
pub trait Network: Send + Debug + Clone + 'static {
    type Tensor: R2lTensor;

    fn input_shape(&self) -> Shape;
    fn output_shape(&self) -> Shape;
    fn io_shape(&self) -> (Shape, Shape) {
        (self.input_shape(), self.output_shape())
    }

    /// Evaluates a nonempty batch, preserving gradients through network parameters.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible input dimensions or failed evaluation.
    fn forward(&self, t: Self::Tensor) -> Result<Self::Tensor>;
}
