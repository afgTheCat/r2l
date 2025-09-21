use bincode::{
    Decode, Encode,
    de::{self},
    error::{DecodeError, EncodeError},
};
use candle_core::{Device, Error, Result, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{Activation, Linear, Module, VarBuilder, linear};
use either::Either;
use safetensors::serialize;

#[derive(Debug, Clone)]
pub struct LinearLayer {
    layer: Linear,
    weight_name: String,
    bias_name: String,
}

impl LinearLayer {
    pub fn new(in_dim: usize, out_dim: usize, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let weight_name = format!("{prefix}_weight");
        let bias_name = format!("{prefix}_bias");
        let layer = linear(in_dim, out_dim, vb.pp(prefix))?;
        Ok(Self {
            layer,
            weight_name,
            bias_name,
        })
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let data = [
            (&self.weight_name, self.layer.weight()),
            (&self.bias_name, self.layer.bias().unwrap()), // TODO: maybe error handle here?
        ];
        serialize(data, &None).map_err(Error::wrap)
    }
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.layer.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct ActivationLayer(pub Activation);

impl Module for ActivationLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct ThreadSafeLayer(pub Either<LinearLayer, ActivationLayer>);

impl ThreadSafeLayer {
    pub fn linear(linear: LinearLayer) -> Self {
        Self(Either::Left(linear))
    }

    pub fn activation(activation: ActivationLayer) -> Self {
        Self(Either::Right(activation))
    }
}

impl Module for ThreadSafeLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.0 {
            Either::Left(linear) => linear.forward(xs),
            Either::Right(activation) => activation.forward(xs),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct ThreadSafeSequential {
    layers: Vec<ThreadSafeLayer>,
}

impl Module for ThreadSafeSequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

pub fn build_sequential(
    input_dim: usize,
    layers: &[usize],
    vb: &VarBuilder,
    prefix: &str,
) -> Result<ThreadSafeSequential> {
    let mut last_dim = input_dim;
    let mut nn = ThreadSafeSequential::default();
    let num_layers = layers.len();
    for (layer_idx, layer_size) in layers.iter().enumerate() {
        let layer_pp = format!("{prefix}{layer_idx}");
        if layer_idx == num_layers - 1 {
            let layer = LinearLayer::new(last_dim, *layer_size, vb, &layer_pp)?;
            nn = nn.add_layer(ThreadSafeLayer::linear(layer))
        } else {
            let lin_layer = LinearLayer::new(last_dim, *layer_size, vb, &layer_pp)?;
            nn = nn.add_layer(ThreadSafeLayer::linear(lin_layer)).add_layer(
                ThreadSafeLayer::activation(ActivationLayer(Activation::Relu)),
            );
        }
        last_dim = *layer_size;
    }
    Ok(nn)
}

impl ThreadSafeSequential {
    pub fn add_layer(mut self, layer: ThreadSafeLayer) -> Self {
        self.layers.push(layer);
        self
    }
}
