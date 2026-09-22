use burn::nn::{
    Dropout,
    activation::Activation,
    conv::Conv2d,
    pool::{AvgPool2d, MaxPool2d},
};
use burn::{module::Module, prelude::Backend, tensor::Tensor};

use crate::networks::Network;
use crate::networks::mlp::Mlp;

#[derive(Module, Debug)]
enum CNNLayer<B: Backend> {
    Activation(Activation<B>),
    Conv(Conv2d<B>),
    MaxPool(MaxPool2d),
    AvgPool(AvgPool2d),
    Dropout(Dropout),
}

impl<B: Backend> CNNLayer<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self {
            CNNLayer::Activation(activation) => activation.forward(input),
            CNNLayer::Conv(conv2d) => conv2d.forward(input),
            CNNLayer::MaxPool(max_pool2d) => max_pool2d.forward(input),
            CNNLayer::AvgPool(avg_pool2d) => avg_pool2d.forward(input),
            CNNLayer::Dropout(dropout) => dropout.forward(input),
        }
    }
}

#[derive(Module, Debug)]
pub struct Cnn<B: Backend> {
    /// Shape of one observation: [channels, height, width].
    shape: [usize; 3],
    cnn_layers: Vec<CNNLayer<B>>,
    mlp: Mlp<B>,
}

impl<B: Backend> Cnn<B> {
    fn forward_inner(&self, mut t: Tensor<B, 4>) -> Tensor<B, 2> {
        for layer in &self.cnn_layers {
            t = layer.forward(t);
        }
        let t: Tensor<B, 2> = t.flatten(1, 3);
        self.mlp.forward(t)
    }
}

impl<B: Backend> Network for Cnn<B> {
    type Tensor = Tensor<B, 2>;

    fn forward(&self, t: Self::Tensor) -> Self::Tensor {
        let [channels, height, width] = self.shape;
        let batch_size = t.dims()[0];
        let t: Tensor<B, 4> = t.reshape([batch_size, channels, height, width]);
        self.forward_inner(t)
    }
}
