use std::fmt::Debug;

use r2l_core::{Shape, error::Result, tensor::R2lTensor};

pub mod burn;
pub mod candle;
pub use candle::seeded_var_builder;

/// A network mapping flat observations to flat outputs.
///
/// Input and output shapes describe one batch element: `[input_size]` and
/// `[output_size]`, both with positive widths. Forwarding accepts
/// `[batch, input_size]` and returns `[batch, output_size]`, preserving the
/// nonzero batch size. CNNs reshape flat inputs internally using their spatial shape.
/// Declared shapes must remain stable for the lifetime of the network.
pub trait Network: Send + Debug + Clone + 'static {
    type Tensor: R2lTensor;

    /// Returns an inference-ready network without changing the training instance.
    /// Parameter storage may remain shared with the original.
    #[must_use]
    fn for_inference(&self) -> Self;

    /// Prepares observation or action inputs for this network's execution mode.
    /// Backends that need explicit input detachment during inference override this.
    fn prepare_input(&self, t: Self::Tensor) -> Self::Tensor {
        t
    }

    fn input_shape(&self) -> Shape;
    fn output_shape(&self) -> Shape;
    fn io_shape(&self) -> (Shape, Shape) {
        (self.input_shape(), self.output_shape())
    }

    /// Evaluates a nonempty batch, preserving parameter gradients in training mode.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible input dimensions or failed evaluation.
    fn forward(&self, t: Self::Tensor) -> Result<Self::Tensor>;

    /// Width of the features immediately before the output layer, when exposed by this network.
    fn feature_size(&self) -> Option<usize> {
        None
    }

    /// Evaluates a batch and returns `(output, features)` for state-dependent exploration.
    ///
    /// # Errors
    /// Returns an error if the network does not expose features or evaluation fails.
    fn forward_with_features(&self, _t: Self::Tensor) -> Result<(Self::Tensor, Self::Tensor)> {
        Err(r2l_core::error::Error::invalid_parameter(
            "state-dependent exploration",
            "a network exposing its output-layer features",
            "features are unavailable",
        ))
    }
}
