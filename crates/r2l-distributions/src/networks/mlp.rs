use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::{Dropout, EluConfig, HardSigmoidConfig, LeakyReluConfig, LinearConfig};
use burn::{module::Module, nn::Linear, prelude::Backend, tensor::Tensor};
use r2l_core::{
    Shape,
    error::{Error, Result},
    models::ActivationFunction,
    networks::MlpConfig,
};

use crate::networks::Network;

#[derive(Debug, Module)]
#[allow(clippy::large_enum_variant)]
pub enum LinearLayer<B: Backend> {
    Activation(Activation<B>),
    LinearLayer(Linear<B>),
    Dropout(Dropout),
}

impl<B: Backend> LinearLayer<B> {
    pub fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self {
            Self::LinearLayer(linear) => linear.forward(t),
            Self::Activation(activation) => activation.forward(t),
            Self::Dropout(dropout) => dropout.forward(t),
        }
    }

    pub(super) fn activation(activation: ActivationFunction, device: &B::Device) -> Activation<B> {
        let config = match activation {
            ActivationFunction::Elu => ActivationConfig::Elu(EluConfig::new()),
            ActivationFunction::Gelu => ActivationConfig::Gelu,
            ActivationFunction::GeluApproximate => ActivationConfig::GeluApproximate,
            ActivationFunction::HardSigmoid => {
                ActivationConfig::HardSigmoid(HardSigmoidConfig::new())
            }
            ActivationFunction::HardSwish => ActivationConfig::HardSwish,
            ActivationFunction::LeakyRelu => ActivationConfig::LeakyRelu(LeakyReluConfig::new()),
            ActivationFunction::Relu => ActivationConfig::Relu,
            ActivationFunction::Sigmoid => ActivationConfig::Sigmoid,
            ActivationFunction::Tanh => ActivationConfig::Tanh,
        };
        config.init::<B>(device)
    }

    fn linear(input: usize, output: usize, device: &B::Device) -> Self {
        let liner_config = LinearConfig::new(input, output).with_bias(true);
        let linear: Linear<B> = liner_config.init::<B>(device);
        Self::LinearLayer(linear)
    }
}

#[derive(Debug, Module)]
pub struct Mlp<B: Backend> {
    layers: Vec<LinearLayer<B>>,
    input_size: usize,
    output_size: usize,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, mut t: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in &self.layers {
            t = layer.forward(t);
        }
        t
    }

    /// Builds a dense network from positive input, hidden and output widths.
    ///
    /// # Panics
    /// Panics if fewer than two widths are supplied or any width is zero.
    #[must_use]
    pub fn build(layer_sizes: &[usize], activation: ActivationFunction) -> Self {
        Self::build_on_device(layer_sizes, activation, &Default::default())
    }

    pub(super) fn from_config(
        config: &MlpConfig,
        input_size: usize,
        output_size: usize,
        device: &B::Device,
    ) -> Self {
        let layers = [&[input_size][..], &config.hidden_layers, &[output_size]].concat();
        Self::build_on_device(&layers, config.activation, device)
    }

    fn build_on_device(
        layer_sizes: &[usize],
        activation: ActivationFunction,
        device: &B::Device,
    ) -> Self {
        assert!(
            layer_sizes.len() >= 2 && !layer_sizes.contains(&0),
            "MLP requires positive input and output widths"
        );
        let mut last_dim = layer_sizes[0];
        let mut layers = vec![];
        let num_layers = layer_sizes.len();
        for (layer_idx, layer_size) in layer_sizes.iter().enumerate().skip(1) {
            if layer_idx == num_layers - 1 {
                layers.push(LinearLayer::linear(last_dim, *layer_size, device));
            } else {
                layers.push(LinearLayer::linear(last_dim, *layer_size, device));
                layers.push(LinearLayer::Activation(LinearLayer::activation(
                    activation, device,
                )));
            }
            last_dim = *layer_size;
        }
        Self {
            layers,
            input_size: layer_sizes[0],
            output_size: last_dim,
        }
    }
}

impl<B: Backend> Network for Mlp<B> {
    type Tensor = Tensor<B, 2>;

    fn input_shape(&self) -> Shape {
        [self.input_size].into()
    }

    fn output_shape(&self) -> Shape {
        [self.output_size].into()
    }

    fn forward(&self, t: Self::Tensor) -> Result<Self::Tensor> {
        let [batch_size, features] = t.dims();
        if batch_size == 0 || features != self.input_size {
            return Err(Error::invalid_parameter(
                "network input shape",
                format!("[nonzero batch, {}]", self.input_size),
                format!("{:?}", t.dims()),
            ));
        }
        Ok(self.forward(t))
    }
}
