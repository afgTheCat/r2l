use burn::nn::{
    Dropout,
    activation::Activation,
    conv::Conv2d,
    pool::{AvgPool2d, MaxPool2d},
};
use burn::nn::{
    conv::Conv2dConfig,
    pool::{AvgPool2dConfig, MaxPool2dConfig},
};
use burn::{module::Module, prelude::Backend, tensor::Tensor};
use r2l_core::networks::{CnnConfig, CnnLayerConfig};
use r2l_core::{
    Shape,
    error::{Error, Result},
};

use super::{burn::NetworkKind, mlp::LinearLayer};
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
    pub(super) fn build(
        config: &CnnConfig,
        observation_shape: &Shape,
        output_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let shape: [usize; 3] = observation_shape.dims().try_into().map_err(|_| {
            NetworkKind::<B>::invalid_config(
                "CNN observations must have shape [channels, height, width]",
            )
        })?;
        let [mut channels, mut height, mut width] = shape;
        let mut cnn_layers = Vec::new();
        for layer in &config.layers {
            let (kernel_size, stride) = match layer {
                CnnLayerConfig::Conv2d {
                    kernel_size,
                    stride,
                    ..
                }
                | CnnLayerConfig::MaxPool2d {
                    kernel_size,
                    stride,
                }
                | CnnLayerConfig::AvgPool2d {
                    kernel_size,
                    stride,
                } => (*kernel_size, *stride),
                CnnLayerConfig::Activation(activation) => {
                    cnn_layers.push(CNNLayer::Activation(LinearLayer::activation(
                        *activation,
                        device,
                    )));
                    continue;
                }
            };
            if kernel_size.contains(&0)
                || stride.contains(&0)
                || kernel_size[0] > height
                || kernel_size[1] > width
            {
                return Err(NetworkKind::<B>::invalid_config(
                    "kernel and stride must be positive, and the kernel must fit its input",
                ));
            }
            height = (height - kernel_size[0]) / stride[0] + 1;
            width = (width - kernel_size[1]) / stride[1] + 1;
            let layer = match layer {
                CnnLayerConfig::Conv2d { out_channels, .. } => {
                    if *out_channels == 0 {
                        return Err(NetworkKind::<B>::invalid_config(
                            "convolution channels must be positive",
                        ));
                    }
                    let conv = Conv2dConfig::new([channels, *out_channels], kernel_size)
                        .with_stride(stride)
                        .init(device);
                    channels = *out_channels;
                    CNNLayer::Conv(conv)
                }
                CnnLayerConfig::MaxPool2d { .. } => CNNLayer::MaxPool(
                    MaxPool2dConfig::new(kernel_size)
                        .with_strides(stride)
                        .init(),
                ),
                CnnLayerConfig::AvgPool2d { .. } => CNNLayer::AvgPool(
                    AvgPool2dConfig::new(kernel_size)
                        .with_strides(stride)
                        .init(),
                ),
                CnnLayerConfig::Activation(_) => unreachable!(),
            };
            cnn_layers.push(layer);
        }
        let input_size = NetworkKind::<B>::flat_size(&Shape::from([channels, height, width]))?;
        let mlp = Mlp::from_config(&config.mlp, input_size, output_size, device);
        Ok(Self {
            shape: shape.into(),
            cnn_layers,
            mlp,
        })
    }

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
