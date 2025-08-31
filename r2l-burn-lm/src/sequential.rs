use burn::nn::LinearConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, GradientsParams};
use burn::tensor::activation::relu;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Shape, TensorData};
use burn::{module::Module, nn::Linear, prelude::Backend, tensor::Tensor};

#[derive(Debug, Clone, Module)]
pub struct ReluAct;

#[derive(Debug, Module)]
enum SequentialLayer<B: Backend> {
    Activation(ReluAct),
    LinearLayer(Linear<B>),
}

impl<B: Backend> SequentialLayer<B> {
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
    layers: Vec<SequentialLayer<B>>,
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
                layers.push(SequentialLayer::layer(last_dim, *layer_size));
            } else {
                layers.push(SequentialLayer::layer(last_dim, *layer_size));
                layers.push(SequentialLayer::relu_act());
            }
            last_dim = *layer_size;
        }
        Self { layers }
    }
}
