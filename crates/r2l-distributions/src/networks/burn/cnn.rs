use std::num::NonZeroUsize;

use burn::nn::{
    conv::Conv2dConfig,
    pool::{AvgPool2dConfig, MaxPool2dConfig},
};
use burn::{module::Module, tensor::Tensor};
use burn::{
    nn::{
        Dropout,
        activation::Activation,
        conv::Conv2d,
        pool::{AvgPool2d, MaxPool2d},
    },
    tensor::Device,
};
use r2l_core::networks::{CnnConfig, CnnLayerConfig};
use r2l_core::{
    Shape,
    error::{Error, Result},
};

use super::{
    NetworkKind,
    mlp::{LinearLayer, Mlp},
};
use crate::networks::Network;

#[derive(Module, Debug)]
enum CNNLayer {
    Activation(Activation),
    Conv(Conv2d),
    MaxPool(MaxPool2d),
    AvgPool(AvgPool2d),
    Dropout(Dropout),
}

impl CNNLayer {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
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
pub struct Cnn {
    #[module(skip)]
    shape: [NonZeroUsize; 3],
    cnn_layers: Vec<CNNLayer>,
    mlp: Mlp,
}

impl Cnn {
    pub(super) fn build(
        config: &CnnConfig,
        observation_shape: &Shape,
        output_size: NonZeroUsize,
        device: &Device,
    ) -> Result<Self> {
        let shape: [usize; 3] = observation_shape.dims().try_into().map_err(|_| {
            NetworkKind::invalid_config(
                "CNN observations must have shape [channels, height, width]",
            )
        })?;
        let [Some(channels), Some(height), Some(width)] = shape.map(NonZeroUsize::new) else {
            return Err(NetworkKind::invalid_config(
                "CNN dimensions must be positive",
            ));
        };
        let shape = [channels, height, width];
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
            let [
                Some(kernel_height),
                Some(kernel_width),
                Some(stride_height),
                Some(stride_width),
            ] = [kernel_size[0], kernel_size[1], stride[0], stride[1]].map(NonZeroUsize::new)
            else {
                return Err(NetworkKind::invalid_config(
                    "kernel and stride must be positive, and the kernel must fit its input",
                ));
            };
            if kernel_height > height || kernel_width > width {
                return Err(NetworkKind::invalid_config(
                    "kernel and stride must be positive, and the kernel must fit its input",
                ));
            }
            height = NonZeroUsize::new((height.get() - kernel_height.get()) / stride_height + 1)
                .expect("a fitting kernel produces a positive height");
            width = NonZeroUsize::new((width.get() - kernel_width.get()) / stride_width + 1)
                .expect("a fitting kernel produces a positive width");
            let layer = match layer {
                CnnLayerConfig::Conv2d { out_channels, .. } => {
                    let out_channels = NonZeroUsize::new(*out_channels).ok_or_else(|| {
                        NetworkKind::invalid_config("convolution channels must be positive")
                    })?;
                    let conv = Conv2dConfig::new([channels.get(), out_channels.get()], kernel_size)
                        .with_stride(stride)
                        .init(device);
                    channels = out_channels;
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
        let input_size =
            NetworkKind::flat_size(&Shape::from([channels.get(), height.get(), width.get()]))?;
        let mlp = Mlp::from_config(&config.mlp, input_size, output_size, device);
        Ok(Self {
            shape,
            cnn_layers,
            mlp,
        })
    }

    fn convolve(&self, mut t: Tensor<4>) -> Tensor<2> {
        for layer in &self.cnn_layers {
            t = layer.forward(t);
        }
        t.flatten(1, 3)
    }
}

impl Network for Cnn {
    type Tensor = Tensor<2>;

    fn input_shape(&self) -> Shape {
        [self.shape.iter().map(|size| size.get()).product::<usize>()].into()
    }

    fn output_shape(&self) -> Shape {
        self.mlp.output_shape()
    }

    fn forward(&self, t: Self::Tensor) -> Result<Self::Tensor> {
        self.forward_with_features(t).map(|(output, _)| output)
    }

    fn feature_size(&self) -> Option<usize> {
        self.mlp.feature_size()
    }

    fn forward_with_features(&self, t: Self::Tensor) -> Result<(Self::Tensor, Self::Tensor)> {
        let [batch_size, features] = t.dims();
        let input_size = self.shape.iter().map(|size| size.get()).product::<usize>();
        if batch_size == 0 || features == 0 || features != input_size {
            return Err(Error::invalid_parameter(
                "network input shape",
                format!("[nonzero batch, {input_size}]"),
                format!("{:?}", t.dims()),
            ));
        }
        let [channels, height, width] = self.shape.map(NonZeroUsize::get);
        let t: Tensor<4> = t.reshape([batch_size, channels, height, width]);
        self.mlp.forward_with_features(self.convolve(t))
    }
}
