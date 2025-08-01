use burn::{
    module::Module,
    nn::Linear,
    prelude::Backend,
    tensor::{Tensor, activation::relu, module::linear},
};

#[derive(Module, Debug)]
pub struct ThreadSafeLinear<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> ThreadSafeLinear<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        linear(
            input,
            self.weight.clone(),
            self.bias.as_ref().map(|b| b.clone()),
        )
    }
}

#[derive(Debug, Clone, Module)]
pub struct ReluAct;

#[derive(Debug)]
enum ThreadSafeLayer<B: Backend> {
    Activation(ReluAct),
    Layer(ThreadSafeLinear<B>),
}

impl<B: Backend> ThreadSafeLayer<B> {
    fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self {
            Self::Layer(linear) => linear.forward(t),
            Self::Activation(ReluAct) => relu(t),
        }
    }
}

#[derive(Debug)]
pub struct ThreadSafeSequential<B: Backend> {
    layers: Vec<ThreadSafeLayer<B>>,
}

impl<B: Backend> ThreadSafeSequential<B> {
    pub fn forward(&self, mut t: Tensor<B, 2>) -> Tensor<B, 2> {
        for layer in self.layers.iter() {
            t = layer.forward(t)
        }
        t
    }
}

#[derive(Debug, Module)]
enum SequentialLayer<B: Backend> {
    Activation(ReluAct),
    Layer(Linear<B>),
}

impl<B: Backend> SequentialLayer<B> {
    fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self {
            Self::Layer(linear) => linear.forward(t),
            Self::Activation(ReluAct) => relu(t),
        }
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
}

#[cfg(test)]
mod test {
    use crate::thread_safe_sequential::{Sequential, SequentialLayer};
    use burn::optim::Optimizer;
    use burn::{
        backend::{Autodiff, NdArray},
        nn::LinearConfig,
        optim::{AdamWConfig, GradientsParams},
        prelude::Backend,
        tensor::{Distribution, Shape, Tensor},
    };

    type MyBackend = Autodiff<NdArray>;

    #[test]
    fn testing_shit() {
        let device = <Autodiff<NdArray> as Backend>::Device::default();
        let liner_config = LinearConfig::new(10, 10).with_bias(true);
        let tens: Tensor<MyBackend, 2> =
            Tensor::random(Shape::new([10, 10]), Distribution::Normal(0., 1.), &device);
        let linear = liner_config.init::<MyBackend>(&device);
        let layer = SequentialLayer::Layer(linear.clone());
        let sequential = Sequential {
            layers: vec![layer],
        };
        let t: Tensor<MyBackend, 2> =
            Tensor::random(Shape::new([10, 10]), Distribution::Normal(0., 1.), &device);
        let s = sequential.forward(t);
        let grads = s.backward();
        let grads =
            GradientsParams::from_grads::<MyBackend, Sequential<MyBackend>>(grads, &sequential);
        let mut opt = AdamWConfig::new().init();
        opt.step(0.01, linear, grads);
    }
}
