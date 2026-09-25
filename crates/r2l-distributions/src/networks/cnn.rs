use burn::nn::{
    Dropout,
    activation::Activation,
    conv::Conv2d,
    pool::{AvgPool2d, MaxPool2d},
};
use burn::{module::Module, prelude::Backend, tensor::Tensor};
use r2l_core::{
    Shape,
    error::{Error, Result},
};

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
    #[module(skip)]
    shape: Shape,
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

    fn input_shape(&self) -> Shape {
        [self.shape.num_elements()].into()
    }

    fn output_shape(&self) -> Shape {
        self.mlp.output_shape()
    }

    fn forward(&self, t: Self::Tensor) -> Result<Self::Tensor> {
        let [batch_size, features] = t.dims();
        let input_size = self.shape.num_elements();
        if batch_size == 0 || features == 0 || features != input_size {
            return Err(Error::invalid_parameter(
                "network input shape",
                format!("[nonzero batch, {input_size}]"),
                format!("{:?}", t.dims()),
            ));
        }
        let [channels, height, width] = *self.shape.dims() else {
            unreachable!("CNN input shape must have three dimensions");
        };
        let t: Tensor<B, 4> = t.reshape([batch_size, channels, height, width]);
        Ok(self.forward_inner(t))
    }
}
