use burn::{Tensor, module::Module, tensor::backend::Backend};

pub(crate) mod cnn;
pub(crate) mod mlp;

/// Maps flat observations to network outputs, handling any spatial reshaping internally.
pub trait Network<B: Backend>: Module<B> + 'static {
    /// Processes one observation and returns exactly one row: `[1, output_size]`.
    ///
    /// # Arguments
    ///
    /// * `t` - A flat observation of shape `[observation_size]`, in the layout
    ///   expected by the network.
    fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2>;

    /// Processes a batch and returns `[t.len(), output_size]`.
    ///
    /// Output row `i` corresponds to observation `t[i]`; input order is preserved.
    /// The output size is the same as for single-observation forwarding.
    ///
    /// # Arguments
    ///
    /// * `t` - A non-empty slice of flat observations, each of shape
    ///   `[observation_size]` and in the layout expected by the network.
    fn batch_forward(&self, t: &[Tensor<B, 1>]) -> Tensor<B, 2>;
}
