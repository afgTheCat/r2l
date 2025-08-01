use burn::{
    optim::{AdamW, SimpleOptimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};

use crate::{
    distributions::DistribnutionKind, policies::Policy,
    thread_safe_sequential::ThreadSafeSequential,
};

struct ParallModel {}

pub struct ParalellActorCritic<B: AutodiffBackend> {
    distribution: DistribnutionKind<B>,
    value_net: ThreadSafeSequential<B>,
    // optimizer: OptimizerAdaptor<AdamW>,
}

impl<B: AutodiffBackend> Policy<B> for ParalellActorCritic<B> {
    type Dist = DistribnutionKind<B>;

    fn distribution(&self) -> &Self::Dist {
        &self.distribution
    }

    fn update(
        &mut self,
        policy_loss: burn::prelude::Tensor<B, 2>,
        value_loss: burn::prelude::Tensor<B, 2>,
    ) {
        let loss = policy_loss + value_loss;
        // let grads = loss.backward();
        // self.optimizer.step(0.01, loss, grads, None);
        // todo!()
    }
}

#[cfg(test)]
mod test {
    use burn::{
        nn::LinearConfig,
        optim::{AdamWConfig, GradientsParams, Optimizer},
        tensor::{Distribution, Shape, Tensor, backend::AutodiffBackend},
    };

    // example on how we use the backwards thing
    fn sequential_test<B: AutodiffBackend>() {
        let device = B::Device::default();
        let liner_config = LinearConfig::new(10, 10).with_bias(true);
        let t: Tensor<B, 2> =
            Tensor::random(Shape::new([10, 10]), Distribution::Normal(0., 1.), &device);
        let linear = liner_config.init::<B>(&device);
        let s = linear.forward(t);
        let grads = s.backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let mut opt = AdamWConfig::new().init();
        opt.step(0.01, linear, grads);
    }
}
