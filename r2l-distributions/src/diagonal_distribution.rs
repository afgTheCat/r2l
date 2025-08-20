use burn::{
    prelude::Backend,
    tensor::{Distribution as TDistribution, Tensor, cast::ToElement},
};
use core::f32;
use r2l_core2::{
    distributions::Distribution,
    env::{Action, Logp, Observation},
    thread_safe_sequential::ThreadSafeSequential,
    utils::tensor_sqr,
};

#[derive(Debug, Clone)]
pub struct DiagGaussianDistribution<B: Backend> {
    noise: Tensor<B, 2>, // TODO:
    log_std: Tensor<B, 2>,
    mu_net: ThreadSafeSequential<B>,
    device: B::Device,
}

impl<B: Backend> DiagGaussianDistribution<B> {
    fn logp_from_t(&self, states: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 1> {
        let mu = self.mu_net.forward(states);
        let std = self.log_std.clone().exp();
        let var = tensor_sqr(std);
        let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
        let log_probs: Tensor<B, 2> = tensor_sqr(actions - mu) / (2. * var);
        let log_probs: Tensor<B, 2> = log_probs.neg();
        let log_probs: Tensor<B, 2> = log_probs - self.log_std.clone() - log_sqrt_2pi;
        log_probs.sum()
    }
}

impl<B: Backend> Distribution for DiagGaussianDistribution<B> {
    type Observation = Tensor<B, 2>;
    type Action = Tensor<B, 2>;
    type Logp = Tensor<B, 2>;

    fn get_action(&self, observation: Self::Observation) -> (Self::Action, f32) {
        let mu = self.mu_net.forward(observation.clone());
        let std = self.log_std.clone().exp();
        let noise = Tensor::<B, 2>::random(
            self.log_std.shape(),
            TDistribution::Normal(0., 1.),
            &self.device,
        );
        let action = mu + std * noise;
        let logp = self.logp_from_t(observation, action.clone());
        (action.squeeze(0), logp.into_scalar().to_f32())
    }

    // TODO: it is a bit wasetful to send the state twice through the nn
    fn log_probs(
        &self,
        observations: &[Self::Observation],
        actions: &[Self::Action],
    ) -> Vec<Self::Logp> {
        let observations = observations.iter().map(|o| o.clone()).collect();
        let observations: Tensor<B, 2> = Tensor::stack(observations, 0);
        let actions = actions.iter().map(|act| act.clone()).collect::<Vec<_>>();
        let actions: Tensor<B, 2> = Tensor::stack(actions, 0);
        let logps = self.logp_from_t(observations, actions);
        todo!()
    }

    fn entropy(&self) -> f32 {
        let log_2pi_plus_1_div_2 = 0.5 * ((2. * f32::consts::PI).ln() + 1.);
        let logp = (self.log_std.clone() + log_2pi_plus_1_div_2).sum();
        logp.into_scalar().to_f32()
    }

    fn std(&self) -> f32 {
        todo!()
    }

    fn resample_noise(&mut self) {
        todo!()
    }
}
