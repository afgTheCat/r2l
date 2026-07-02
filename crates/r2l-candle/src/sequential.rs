use candle_core::{Result, Tensor};
use candle_nn::init::{FanInOut, NonLinearity, NormalOrUniform};
use candle_nn::{Activation, Init, Linear, Module, VarBuilder};
use either::Either;
use r2l_core::models::ActivationFunction;
use safetensors::serialize as st_serialize;
use serde::Serialize;

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

    fn input_size(&self) -> usize {
        let shape = self.layer.weight().shape();
        debug_assert!(shape.rank() == 2);
        shape.dims()[1]
    }

    fn serialize(&self, name: &str) -> Vec<u8> {
        let weight_name = format!("weight_{name}");
        let bias_name = format!("bias_{name}");
        let mut tensors = vec![(weight_name, self.layer.weight())];
        if let Some(bias) = self.layer.bias() {
            tensors.push((bias_name, bias));
        }
        st_serialize(tensors, None).expect("failed to serialize linear layer")
    }
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.layer.forward(xs)
    }
}

#[derive(Debug, Clone, Serialize)]
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

    pub(crate) fn input_size(&self) -> Option<usize> {
        match &self.0 {
            Either::Left(linear) => Some(linear.input_size()),
            Either::Right(_) => None,
        }
    }

    fn serialize(&self, name: &str) -> Vec<u8> {
        match &self.0 {
            Either::Left(linear) => {
                let mut bytes = vec![0];
                bytes.extend(linear.serialize(name));
                bytes
            }
            Either::Right(activation) => {
                let activation_id = match activation.0 {
                    ActivationFunction::Elu => 0,
                    ActivationFunction::Gelu => 1,
                    ActivationFunction::GeluApproximate => 2,
                    ActivationFunction::HardSigmoid => 3,
                    ActivationFunction::HardSwish => 4,
                    ActivationFunction::LeakyRelu => 5,
                    ActivationFunction::Relu => 6,
                    ActivationFunction::Sigmoid => 7,
                    ActivationFunction::Tanh => 8,
                };
                vec![1, activation_id]
            }
        }
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

impl Serialize for Sequential {
    fn serialize<S>(&self, serializer: S) -> std::prelude::v1::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        todo!()
    }
}

impl Sequential {
    fn serialize(&self) -> Vec<u8> {
        let mut serialized = vec![];
        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_name = format!("layer_{idx}");
            let layer_serialized = layer.serialize(&layer_name);
            serialized.extend(layer_serialized);
        }
        serialized
    }

    pub(crate) fn layer(&self, idx: usize) -> Option<&Layer> {
        self.layers.get(idx)
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

impl Sequential {
    fn add_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }
}
