use burn::{Tensor, module::Module, tensor::backend::Backend};
use r2l_core::Shape;
use r2l_core::{
    error::{Error, Result},
    networks::NetworkConfig,
};

pub(crate) mod cnn;
pub(crate) mod mlp;

/// Burn network selected by a backend-independent architecture configuration.
#[derive(Debug, Module)]
pub enum NetworkKind<B: Backend> {
    /// Fully connected network.
    Mlp(mlp::Mlp<B>),
    /// Convolutional network with a dense output network.
    Cnn(cnn::Cnn<B>),
}

impl<B: Backend> From<mlp::Mlp<B>> for NetworkKind<B> {
    fn from(network: mlp::Mlp<B>) -> Self {
        Self::Mlp(network)
    }
}

impl<B: Backend> NetworkKind<B> {
    /// Builds a network for one observation shape and a requested output width.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions or incompatible spatial layers.
    pub fn build(
        config: &NetworkConfig,
        observation_shape: &Shape,
        output_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let input_size = Self::flat_size(observation_shape)?;
        let mlp = match config {
            NetworkConfig::Mlp(config) => config,
            NetworkConfig::Cnn(config) => &config.mlp,
        };
        if output_size == 0 || mlp.hidden_layers.contains(&0) {
            return Err(Self::invalid_config("layer widths must be positive"));
        }
        Ok(match config {
            NetworkConfig::Mlp(config) => Self::Mlp(mlp::Mlp::from_config(
                config,
                input_size,
                output_size,
                device,
            )),
            NetworkConfig::Cnn(config) => Self::Cnn(cnn::Cnn::build(
                config,
                observation_shape,
                output_size,
                device,
            )?),
        })
    }

    pub(super) fn flat_size(shape: &Shape) -> Result<usize> {
        let size = shape.num_elements();
        if size == 0 {
            return Err(Self::invalid_config("shape must have a positive size"));
        }
        Ok(size)
    }

    pub(super) fn invalid_config(details: &str) -> Error {
        Error::invalid_parameter(
            "network config",
            "compatible, positive network dimensions",
            details,
        )
    }
}

impl<B: Backend> Network<B> for NetworkKind<B> {
    fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        match self {
            Self::Mlp(network) => Network::forward(network, t),
            Self::Cnn(network) => network.forward(t),
        }
    }

    fn batch_forward(&self, t: &[Tensor<B, 1>]) -> Tensor<B, 2> {
        match self {
            Self::Mlp(network) => network.batch_forward(t),
            Self::Cnn(network) => network.batch_forward(t),
        }
    }
}

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
