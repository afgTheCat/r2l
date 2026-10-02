//! Burn networks constructed from shared architecture configurations.
use std::num::NonZeroUsize;

use burn::{module::Module, prelude::Backend, tensor::Tensor};
use r2l_core::Shape;
use r2l_core::{
    error::{Error, Result},
    networks::NetworkConfig,
};

use super::{
    Network,
    cnn::{self, Cnn},
    mlp::{self, Mlp},
};

#[derive(Debug, Module)]
pub enum NetworkKind<B: Backend> {
    /// Fully connected network.
    Mlp(Mlp<B>),
    /// Convolutional network with a dense output network.
    Cnn(Cnn<B>),
}

impl<B: Backend> From<Mlp<B>> for NetworkKind<B> {
    fn from(network: Mlp<B>) -> Self {
        Self::Mlp(network)
    }
}

impl<B: Backend> Network for NetworkKind<B> {
    type Tensor = Tensor<B, 2>;

    fn input_shape(&self) -> r2l_core::Shape {
        match self {
            Self::Mlp(mlp) => mlp.input_shape(),
            Self::Cnn(cnn) => cnn.input_shape(),
        }
    }

    fn output_shape(&self) -> r2l_core::Shape {
        match self {
            Self::Mlp(mlp) => mlp.output_shape(),
            Self::Cnn(cnn) => cnn.output_shape(),
        }
    }

    fn forward(&self, t: Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        match self {
            Self::Mlp(mlp) => Network::forward(mlp, t),
            Self::Cnn(cnn) => cnn.forward(t),
        }
    }

    fn feature_size(&self) -> Option<usize> {
        match self {
            Self::Mlp(mlp) => mlp.feature_size(),
            Self::Cnn(cnn) => cnn.feature_size(),
        }
    }

    fn forward_with_features(&self, t: Self::Tensor) -> Result<(Self::Tensor, Self::Tensor)> {
        match self {
            Self::Mlp(mlp) => mlp.forward_with_features(t),
            Self::Cnn(cnn) => cnn.forward_with_features(t),
        }
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
        let output_size = NonZeroUsize::new(output_size)
            .ok_or_else(|| Self::invalid_config("layer widths must be positive"))?;
        if mlp.hidden_layers.contains(&0) {
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

    pub(super) fn flat_size(shape: &Shape) -> Result<NonZeroUsize> {
        NonZeroUsize::new(shape.num_elements())
            .ok_or_else(|| Self::invalid_config("shape must have a positive size"))
    }

    pub(super) fn invalid_config(details: &str) -> Error {
        Error::invalid_parameter(
            "network config",
            "compatible, positive network dimensions",
            details,
        )
    }
}
