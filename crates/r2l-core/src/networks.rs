//! Backend-independent network architecture configurations.

use serde::{Deserialize, Serialize};

use crate::models::ActivationFunction;

/// Architecture of a network with a linear output layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkConfig {
    /// Fully connected network over flattened observations.
    Mlp(MlpConfig),
    /// Spatial layers followed by a fully connected network.
    Cnn(CnnConfig),
}

impl NetworkConfig {
    /// Returns the dense-layer configuration, including the dense layers after a CNN.
    pub fn mlp_config_mut(&mut self) -> &mut MlpConfig {
        match self {
            Self::Mlp(config) => config,
            Self::Cnn(config) => &mut config.mlp,
        }
    }
}

/// Hidden layers of a fully connected network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    /// Hidden-layer widths, excluding input and output widths. May be empty.
    pub hidden_layers: Vec<usize>,
    /// Activation after each hidden layer; the output layer is linear.
    pub activation: ActivationFunction,
}

/// CNN over observations laid out as `[channels, height, width]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnConfig {
    /// Spatial layers in execution order, with explicit activation layers.
    pub layers: Vec<CnnLayerConfig>,
    /// Dense layers applied after flattening the spatial output.
    pub mlp: MlpConfig,
}

/// Spatial layer configuration. Convolution and pooling use no padding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CnnLayerConfig {
    /// Convolution with a bias and unit dilation.
    Conv2d {
        /// Number of output channels; input channels are inferred.
        out_channels: usize,
        /// Kernel height and width.
        kernel_size: [usize; 2],
        /// Vertical and horizontal strides.
        stride: [usize; 2],
    },
    /// Maximum pooling.
    MaxPool2d {
        /// Kernel height and width.
        kernel_size: [usize; 2],
        /// Vertical and horizontal strides.
        stride: [usize; 2],
    },
    /// Average pooling.
    AvgPool2d {
        /// Kernel height and width.
        kernel_size: [usize; 2],
        /// Vertical and horizontal strides.
        stride: [usize; 2],
    },
    /// Elementwise activation.
    Activation(ActivationFunction),
}
