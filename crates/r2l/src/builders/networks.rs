//! Construct networks independently of policy and value-function configuration.

use burn::prelude::Backend;
use candle_nn::VarBuilder;
use r2l_core::Shape;
use r2l_core::error::{Error, Result};
pub use r2l_core::networks::{CnnConfig, CnnLayerConfig, MlpConfig, NetworkConfig};
use r2l_distributions::networks::candle_mlp::Mlp;
pub use r2l_distributions::{
    Network as BurnNetwork, networks::burn::NetworkKind as BurnNetworkKind,
};

/// Builds a network with observation and output dimensions supplied by its caller.
#[derive(Debug, Clone)]
pub struct NetworkBuilder {
    config: NetworkConfig,
}

impl NetworkBuilder {
    /// Creates a builder without initializing network parameters.
    ///
    /// # Arguments
    ///
    /// * `config` - Hidden architecture; input and output dimensions are supplied at build time.
    #[must_use]
    pub fn new(config: NetworkConfig) -> Self {
        Self { config }
    }

    /// Builds a Burn network with a linear output layer.
    ///
    /// # Arguments
    ///
    /// * `observation_shape` - Shape of one observation, excluding the batch dimension.
    ///   CNNs require `[channels, height, width]`; MLPs flatten all dimensions.
    /// * `output_size` - Number of logits, means, or values produced per observation.
    /// * `device` - Device on which network parameters are initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if layer dimensions or the observation shape are invalid.
    pub fn build_burn<B: Backend>(
        &self,
        observation_shape: &Shape,
        output_size: usize,
        device: &B::Device,
    ) -> Result<BurnNetworkKind<B>> {
        BurnNetworkKind::build(&self.config, observation_shape, output_size, device)
    }

    /// Builds a Candle MLP with a linear output layer.
    ///
    /// # Arguments
    /// * `observation_shape` - Shape of one observation; MLP inputs are flattened.
    /// * `output_size` - Number of network outputs.
    /// * `var_builder` - Parameter source, namespace and device.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions or unsupported CNN configurations.
    pub fn build_candle(
        &self,
        observation_shape: &Shape,
        output_size: usize,
        var_builder: &VarBuilder<'_>,
    ) -> Result<Mlp> {
        let NetworkConfig::Mlp(config) = &self.config else {
            return Err(Error::Unsupported {
                operation: "build Candle CNN".into(),
                details: "Candle supports MLP networks".into(),
            });
        };
        let widths = [
            &[observation_shape.num_elements()][..],
            &config.hidden_layers,
            &[output_size],
        ]
        .concat();
        Mlp::build(&widths, config.activation, var_builder)
    }
}
