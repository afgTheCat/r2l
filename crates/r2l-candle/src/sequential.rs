use std::collections::HashMap;

use candle_core::{Result, Tensor};
use candle_nn::init::{FanInOut, NonLinearity, NormalOrUniform};
use candle_nn::{Activation, Init, Linear, Module, VarBuilder};
use either::Either;
use r2l_core::models::ActivationFunction;

#[derive(Debug, Clone)]
struct LinearLayer {
    layer: Linear,
}

impl LinearLayer {
    fn new(in_dim: usize, out_dim: usize, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let layer_vb = vb.pp(prefix);
        let weight = layer_vb.get_with_hints(
            (out_dim, in_dim),
            "weight",
            Init::Kaiming {
                dist: NormalOrUniform::Uniform,
                fan: FanInOut::FanIn,
                non_linearity: NonLinearity::ExplicitGain(1.0 / 3.0f64.sqrt()),
            },
        )?;
        let bound = 1. / (in_dim as f64).sqrt();
        let bias = layer_vb.get_with_hints(
            out_dim,
            "bias",
            Init::Uniform {
                lo: -bound,
                up: bound,
            },
        )?;
        let layer = Linear::new(weight, Some(bias));
        Ok(Self { layer })
    }

    fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut tensors = vec![(format!("{prefix}.weight"), self.layer.weight().clone())];
        if let Some(bias) = self.layer.bias() {
            tensors.push((format!("{prefix}.bias"), bias.clone()));
        }
        tensors
    }
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.layer.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct ActivationLayer(ActivationFunction);

impl ActivationLayer {
    fn new(activation: ActivationFunction) -> Self {
        Self(activation)
    }
}

impl Module for ActivationLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self.0 {
            ActivationFunction::Elu => Activation::Elu(1.0).forward(xs),
            ActivationFunction::Gelu => Activation::Gelu.forward(xs),
            ActivationFunction::GeluApproximate => Activation::GeluPytorchTanh.forward(xs),
            ActivationFunction::HardSigmoid => Activation::HardSigmoid.forward(xs),
            ActivationFunction::HardSwish => Activation::HardSwish.forward(xs),
            ActivationFunction::LeakyRelu => Activation::LeakyRelu(0.01).forward(xs),
            ActivationFunction::Relu => xs.relu(),
            ActivationFunction::Sigmoid => Activation::Sigmoid.forward(xs),
            ActivationFunction::Tanh => xs.tanh(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Layer(Either<LinearLayer, ActivationLayer>);

impl Layer {
    fn linear(linear: LinearLayer) -> Self {
        Self(Either::Left(linear))
    }

    fn activation(activation: ActivationLayer) -> Self {
        Self(Either::Right(activation))
    }
}

impl Module for Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.0 {
            Either::Left(linear) => linear.forward(xs),
            Either::Right(activation) => activation.forward(xs),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub(crate) struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    pub(crate) fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.layers
            .iter()
            .filter_map(|layer| match &layer.0 {
                Either::Left(linear) => Some(linear),
                Either::Right(_) => None,
            })
            .enumerate()
            .flat_map(|(idx, linear)| linear.named_tensors(&format!("{prefix}{idx}")))
            .collect()
    }
}

impl Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

pub(crate) fn build_sequential(
    input_dim: usize,
    layers: &[usize],
    vb: &VarBuilder,
    prefix: &str,
    activation: ActivationFunction,
) -> Result<Sequential> {
    let mut last_dim = input_dim;
    let mut nn = Sequential::default();
    let num_layers = layers.len();
    for (layer_idx, layer_size) in layers.iter().enumerate() {
        let layer_pp = format!("{prefix}{layer_idx}");
        if layer_idx == num_layers - 1 {
            let layer = LinearLayer::new(last_dim, *layer_size, vb, &layer_pp)?;
            nn = nn.add_layer(Layer::linear(layer))
        } else {
            let lin_layer = LinearLayer::new(last_dim, *layer_size, vb, &layer_pp)?;
            nn = nn
                .add_layer(Layer::linear(lin_layer))
                .add_layer(Layer::activation(ActivationLayer::new(activation)));
        }
        last_dim = *layer_size;
    }
    Ok(nn)
}

pub(crate) fn network_shape(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
) -> (usize, Vec<usize>) {
    let first_weight = tensors.get(&format!("{prefix}0.weight")).unwrap();
    let first_dims = first_weight.dims();

    let observation_size = first_dims[1];
    let mut layers = Vec::new();

    for layer_idx in 0.. {
        let Some(weight) = tensors.get(&format!("{prefix}{layer_idx}.weight")) else {
            break;
        };
        let dims = weight.dims();
        layers.push(dims[0]);
    }

    (observation_size, layers)
}

impl Sequential {
    fn add_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }
}
