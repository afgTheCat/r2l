use candle_core::{Result, Tensor};
use candle_nn::init::{FanInOut, NonLinearity, NormalOrUniform};
use candle_nn::{Activation, Init, Linear, Module, VarBuilder};
use either::Either;

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
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.layer.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct ActivationLayer(Activation);

impl Module for ActivationLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ThreadSafeLayer(Either<LinearLayer, ActivationLayer>);

impl ThreadSafeLayer {
    fn linear(linear: LinearLayer) -> Self {
        Self(Either::Left(linear))
    }

    fn activation(activation: ActivationLayer) -> Self {
        Self(Either::Right(activation))
    }

    pub(crate) fn input_size(&self) -> Option<usize> {
        match &self.0 {
            Either::Left(linear) => Some(linear.input_size()),
            Either::Right(act) => None,
        }
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
pub(crate) struct ThreadSafeSequential {
    layers: Vec<ThreadSafeLayer>,
}

impl ThreadSafeSequential {
    pub(crate) fn layer(&self, idx: usize) -> Option<&ThreadSafeLayer> {
        self.layers.get(idx)
    }
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

pub(crate) fn build_sequential(
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
    fn add_layer(mut self, layer: ThreadSafeLayer) -> Self {
        self.layers.push(layer);
        self
    }
}
