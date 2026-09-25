//! Dense Candle networks with parameters supplied by a native `VarBuilder`.

use candle_core::Tensor;
use candle_nn::{
    Activation, Init, Linear, Module, VarBuilder,
    init::{FanInOut, NonLinearity, NormalOrUniform},
};
use r2l_core::{
    Shape,
    error::{Error, Result},
    models::ActivationFunction,
};

use super::Network;

/// A dense network with an activation between hidden layers and a linear output.
///
/// Build from a `VarMap`-backed builder to register trainable parameters. Clones
/// share parameter storage, so optimizer updates are visible through every clone.
#[derive(Clone, Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
    activation: ActivationFunction,
    input_size: usize,
    output_size: usize,
}

impl Mlp {
    /// Builds a network from positive input, hidden and output widths.
    ///
    /// `vb` controls the device, dtype, initialization and parameter-name prefix.
    /// Layers are registered under `0`, `1`, etc. Use different prefixes for
    /// independent policy and value networks sharing a variable map.
    ///
    /// # Errors
    /// Returns an error for invalid widths or failed parameter initialization.
    pub fn build(
        layer_sizes: &[usize],
        activation: ActivationFunction,
        vb: &VarBuilder<'_>,
    ) -> Result<Self> {
        if layer_sizes.len() < 2 || layer_sizes.contains(&0) {
            return Err(Error::invalid_parameter(
                "layer_sizes",
                "at least two positive widths",
                format!("{layer_sizes:?}"),
            ));
        }
        let layers = layer_sizes
            .windows(2)
            .enumerate()
            .map(|(index, widths)| {
                let vb = vb.pp(index);
                let weight = vb.get_with_hints(
                    (widths[1], widths[0]),
                    "weight",
                    Init::Kaiming {
                        dist: NormalOrUniform::Uniform,
                        fan: FanInOut::FanIn,
                        non_linearity: NonLinearity::ExplicitGain(1.0 / 3.0f64.sqrt()),
                    },
                )?;
                let bound = 1.0 / (widths[0] as f64).sqrt();
                let bias = vb.get_with_hints(
                    widths[1],
                    "bias",
                    Init::Uniform {
                        lo: -bound,
                        up: bound,
                    },
                )?;
                Ok(Linear::new(weight, Some(bias)))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            activation,
            input_size: layer_sizes[0],
            output_size: layer_sizes[layer_sizes.len() - 1],
        })
    }

    pub(crate) fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(index, layer)| {
                let mut tensors =
                    vec![(format!("{prefix}.{index}.weight"), layer.weight().clone())];
                if let Some(bias) = layer.bias() {
                    tensors.push((format!("{prefix}.{index}.bias"), bias.clone()));
                }
                tensors
            })
            .collect()
    }

    fn activate(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let activation = match self.activation {
            ActivationFunction::Elu => Activation::Elu(1.0),
            ActivationFunction::Gelu => Activation::Gelu,
            ActivationFunction::GeluApproximate => Activation::GeluPytorchTanh,
            ActivationFunction::HardSigmoid => Activation::HardSigmoid,
            ActivationFunction::HardSwish => Activation::HardSwish,
            ActivationFunction::LeakyRelu => Activation::LeakyRelu(0.01),
            ActivationFunction::Relu => Activation::Relu,
            ActivationFunction::Sigmoid => Activation::Sigmoid,
            ActivationFunction::Tanh => return t.tanh(),
        };
        activation.forward(t)
    }
}

impl Network for Mlp {
    type Tensor = Tensor;

    fn input_shape(&self) -> Shape {
        [self.input_size].into()
    }

    fn output_shape(&self) -> Shape {
        [self.output_size].into()
    }

    fn forward(&self, mut t: Tensor) -> Result<Tensor> {
        if !matches!(t.dims(), [batch, width] if *batch > 0 && *width == self.input_size) {
            return Err(Error::invalid_parameter(
                "network input shape",
                format!("[nonzero batch, {}]", self.input_size),
                format!("{:?}", t.dims()),
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            t = layer.forward(&t)?;
            if index + 1 < self.layers.len() {
                t = self.activate(&t)?;
            }
        }
        Ok(t)
    }
}
