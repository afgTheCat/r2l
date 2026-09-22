//! Construct networks independently of policy and value-function configuration.

use burn::prelude::Backend;
use candle_nn::{Sequential, VarBuilder};
pub use r2l_burn::networks::{Network as BurnNetwork, NetworkKind as BurnNetworkKind};
use r2l_core::error::{Error, Result};
pub use r2l_core::networks::{CnnConfig, CnnLayerConfig, MlpConfig, NetworkConfig};

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
        observation_shape: &[usize],
        output_size: usize,
        device: &B::Device,
    ) -> Result<BurnNetworkKind<B>> {
        BurnNetworkKind::build(&self.config, observation_shape, output_size, device)
    }

    /// Reserves the Candle construction path for these configurations.
    ///
    /// # Arguments
    ///
    /// * `_observation_shape` - Shape of one observation, excluding the batch dimension.
    /// * `_output_size` - Number of outputs produced per observation.
    /// * `_var_builder` - Parameter source and device for the network.
    ///
    /// # Errors
    ///
    /// MLP construction through this standalone entry point is not implemented.
    ///
    /// # Panics
    ///
    /// CNN construction is a placeholder and panics.
    pub fn build_candle(
        &self,
        _observation_shape: &[usize],
        _output_size: usize,
        _var_builder: &VarBuilder<'_>,
    ) -> Result<Sequential> {
        if matches!(self.config, NetworkConfig::Cnn(_)) {
            todo!("Candle CNN network construction");
        }
        Err(Error::Unsupported {
            operation: "build Candle network from config".into(),
            details: "configuration-based Candle network construction is not implemented yet"
                .into(),
        })
    }
}
