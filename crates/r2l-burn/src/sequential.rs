use burn::nn::activation::{Activation, ActivationConfig};
use burn::nn::{EluConfig, HardSigmoidConfig, LeakyReluConfig, LinearConfig};
use burn::{module::Module, nn::Linear, prelude::Backend, tensor::Tensor};
use burn_store::{ModuleStore, SafetensorsStore};
use r2l_core::models::ActivationFunction;

#[derive(Debug, Module)]
pub enum Layer<B: Backend> {
    Activation(Activation<B>),
    LinearLayer(Linear<B>),
}

impl<B: Backend> Layer<B> {
    fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self {
            Self::LinearLayer(linear) => linear.forward(t),
            Self::Activation(activation) => activation.forward(t),
        }
    }

    fn activation(activation: ActivationFunction) -> Self {
        let device = Default::default();
        let config = match activation {
            ActivationFunction::Elu => ActivationConfig::Elu(EluConfig::new()),
            ActivationFunction::Gelu => ActivationConfig::Gelu,
            ActivationFunction::GeluApproximate => ActivationConfig::GeluApproximate,
            ActivationFunction::HardSigmoid => {
                ActivationConfig::HardSigmoid(HardSigmoidConfig::new())
            }
            ActivationFunction::HardSwish => ActivationConfig::HardSwish,
            ActivationFunction::LeakyRelu => ActivationConfig::LeakyRelu(LeakyReluConfig::new()),
            ActivationFunction::Relu => ActivationConfig::Relu,
            ActivationFunction::Sigmoid => ActivationConfig::Sigmoid,
            ActivationFunction::Tanh => ActivationConfig::Tanh,
        };
        Self::Activation(config.init::<B>(&device))
    }

    fn linear(input: usize, output: usize) -> Self {
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

    pub fn build(layer_sizes: &[usize], activation: ActivationFunction) -> Self {
        let mut last_dim = layer_sizes[0];
        let mut layers = vec![];
        let num_layers = layer_sizes.len();
        for (layer_idx, layer_size) in layer_sizes.iter().enumerate().skip(1) {
            if layer_idx == num_layers - 1 {
                layers.push(Layer::linear(last_dim, *layer_size));
            } else {
                layers.push(Layer::linear(last_dim, *layer_size));
                layers.push(Layer::activation(activation));
            }
            last_dim = *layer_size;
        }
        Self { layers }
    }

    pub fn dims_from_store(name: &str, storage: &mut SafetensorsStore) -> Vec<usize> {
        let mut layer_idx = 0;
        let mut mu_layers = vec![];
        while let Ok(Some(layer)) =
            storage.get_snapshot(&format!("{name}.layers.{layer_idx}.LinearLayer.weight"))
        {
            let shape = layer.shape.dims::<2>();
            if layer_idx == 0 {
                mu_layers.extend([shape[0], shape[1]]);
            } else {
                mu_layers.push(shape[1]);
            }
            layer_idx += 2;
        }
        mu_layers
    }
}
