use core::f32;

use burn::{
    prelude::Backend,
    tensor::{Distribution as TDistribution, Tensor},
};

use crate::{
    distributions::Distribution, thread_safe_sequential::ThreadSafeSequential, utils::tensor_sqr,
};

#[derive(Debug)]
pub struct DiagGaussianDistribution<B: Backend> {
    noise: Tensor<B, 2>, // TODO:
    log_std: Tensor<B, 2>,
    mu_net: ThreadSafeSequential<B>,
    device: B::Device,
}

impl<B: Backend> Distribution<B> for DiagGaussianDistribution<B> {
    fn get_action(&self, observation: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let mu = self.mu_net.forward(observation.clone());
        let std = self.log_std.clone().exp();
        let noise = Tensor::<B, 2>::random(
            self.log_std.shape(),
            TDistribution::Normal(0., 1.),
            &self.device,
        );
        let actions = mu + std * noise;
        let logp = self.log_probs(observation, actions.clone());
        (actions, logp)
    }

    // TODO: it is a bit wasetful to send the state twice through the nn
    fn log_probs(&self, states: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 1> {
        let mu = self.mu_net.forward(states);
        let std = self.log_std.clone().exp();
        let var = tensor_sqr(std);
        let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
        let log_probs: Tensor<B, 2> = tensor_sqr(actions - mu) / (2. * var);
        let log_probs: Tensor<B, 2> = log_probs.neg();
        let log_probs: Tensor<B, 2> = log_probs - self.log_std.clone() - log_sqrt_2pi;
        log_probs.sum()
    }

    fn entropy(&self) -> Tensor<B, 1> {
        let log_2pi_plus_1_div_2 = 0.5 * ((2. * f32::consts::PI).ln() + 1.);
        (self.log_std.clone() + log_2pi_plus_1_div_2).sum()
    }

    fn std(&self) -> f32 {
        todo!()
    }

    fn resample_noise(&mut self) {
        todo!()
    }
}
