use burn::nn::LinearConfig;
use burn::tensor::activation::relu;
use burn::tensor::module::linear;
use burn::{module::Module, nn::Linear, prelude::Backend, tensor::Tensor};

#[derive(Debug, Clone, Module)]
pub struct ReluAct;

#[derive(Debug, Module)]
pub enum Layer<B: Backend> {
    Activation(ReluAct),
    LinearLayer(Linear<B>),
}

impl<B: Backend> Layer<B> {
    fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self {
            Self::LinearLayer(linear) => linear.forward(t),
            Self::Activation(ReluAct) => relu(t),
        }
    }

    fn relu_act() -> Self {
        Self::Activation(ReluAct)
    }

    fn layer(input: usize, output: usize) -> Self {
        let device = Default::default();
        let liner_config = LinearConfig::new(input, output).with_bias(true);
        let linear: Linear<B> = liner_config.init::<B>(&device);
        Self::LinearLayer(linear)
    }
}

#[derive(Debug, Module)]
pub struct Sequential<B: Backend> {
    layers: Vec<Layer<B>>,
}

impl<B: Backend> Sequential<B> {
    pub fn forward(&self, mut t: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in self.layers.iter() {
            t = layer.forward(t)
        }
        t
    }

    pub fn build(layer_sizes: &[usize]) -> Self {
        let mut last_dim = layer_sizes[0];
        let mut layers = vec![];
        let num_layers = layer_sizes.len();
        for (layer_idx, layer_size) in layer_sizes.iter().enumerate().skip(1) {
            if layer_idx == num_layers - 1 {
                layers.push(Layer::layer(last_dim, *layer_size));
            } else {
                layers.push(Layer::layer(last_dim, *layer_size));
                layers.push(Layer::relu_act());
            }
            last_dim = *layer_size;
        }
        Self { layers }
    }
}

#[derive(Debug)]
pub struct FrozenLinear<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> FrozenLinear<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        linear(input, self.weight.clone(), self.bias.clone())
    }

    pub fn from_linear(linear: Linear<B>) -> Self {
        Self {
            weight: linear.weight.val(),
            bias: linear.bias.map(|b| b.val()),
        }
    }
}

#[derive(Debug)]
pub enum FrozenLayer<B: Backend> {
    Activation(ReluAct),
    LinearLayer(FrozenLinear<B>),
}

impl<B: Backend> FrozenLayer<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self {
            Self::LinearLayer(linear) => linear.forward(input),
            Self::Activation(ReluAct) => relu(input),
        }
    }

    pub fn from_layer(layer: Layer<B>) -> Self {
        match layer {
            Layer::LinearLayer(linear) => Self::LinearLayer(FrozenLinear::from_linear(linear)),
            Layer::Activation(act) => Self::Activation(act),
        }
    }
}

#[derive(Debug)]
pub struct FrozenSequential<B: Backend> {
    pub layers: Vec<FrozenLayer<B>>,
}

impl<B: Backend> FrozenSequential<B> {
    pub fn forward(&self, mut input: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in self.layers.iter() {
            input = layer.forward(input)
        }
        input
    }

    pub fn from_sequential(sequential: Sequential<B>) -> Self {
        let layers = sequential
            .layers
            .into_iter()
            .map(FrozenLayer::from_layer)
            .collect::<Vec<_>>();
        Self { layers }
    }
}
