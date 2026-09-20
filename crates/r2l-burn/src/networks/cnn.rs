use burn::{module::Module, prelude::Backend, tensor::Tensor};
use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        activation::Activation,
        conv::{Conv2d, Conv2dConfig},
        pool::{AvgPool2d, MaxPool2d, MaxPool2dConfig},
    },
    tensor::Shape,
};

use crate::networks::Network;
use crate::networks::mlp::{LinearLayer, Mlp};

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
    shape: Shape,
    cnn_layers: Vec<CNNLayer<B>>,
    mlp: Mlp<B>,
}

impl<B: Backend> Cnn<B> {
    fn forward_inner(&self, mut t: Tensor<B, 4>) -> Tensor<B, 2> {
        for layer in &self.cnn_layers {
            t = layer.forward(t);
        }
        let mut t: Tensor<B, 2> = t.flatten(1, 3);
        self.mlp.forward(t)
    }
}

impl<B: Backend> Network<B> for Cnn<B> {
    fn batch_forward(&self, t: &[Tensor<B, 1>]) -> Tensor<B, 2> {
        let t: Tensor<B, 4> = Tensor::stack(
            t.iter()
                .map(|t| t.clone().reshape::<3, _>(self.shape.clone()))
                .collect(),
            0,
        );
        self.forward_inner(t)
    }

    fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let t: Tensor<B, 3> = t.reshape(self.shape.clone());
        let t: Tensor<B, 4> = t.unsqueeze();
        self.forward_inner(t)
    }
}
