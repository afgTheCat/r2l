// TODO: burn support is highly experimental, and we need to figure out the right abstractions at
// one point. Maybe next release.
pub mod burn_rollout_buffer;
pub mod distributions;
pub mod learning_module;
pub mod sequential;

use burn::optim::AdamW;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::backend::AutodiffBackend;
use burn::{module::Module, prelude::Backend, tensor::Tensor};
use std::f32;

use crate::sequential::Sequential;

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    pub mu_net: Sequential<B>,
    pub value_net: Sequential<B>,
    pub log_std: Tensor<B, 2>,
    //  TODO: action_low, action_high should not be part of the model!
    pub obs_size: usize,
    pub action_low: Tensor<B, 2>,
    pub action_high: Tensor<B, 2>,
}

struct BurnPPODebug<B: AutodiffBackend> {
    model: Model<B>,
    optimizer: OptimizerAdaptor<AdamW, Model<B>, B>,
}

impl<B: AutodiffBackend> BurnPPODebug<B> {
    fn logp(&self, states: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 1> {
        let device = Default::default();
        let mu = self.model.mu_net.forward(states);
        let std = self.model.log_std.clone().exp();
        let var = std.clone() * std;
        let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
        let log_sqrt_2pi: Tensor<B, 2> = Tensor::full(mu.shape(), log_sqrt_2pi, &device);
        let actions_minus_mu = actions - mu;
        let log_probs: Tensor<B, 2> = (actions_minus_mu.clone() * actions_minus_mu) / (2 * var);
        let log_probs = log_probs.neg() - self.model.log_std.clone() - log_sqrt_2pi;
        log_probs.sum_dim(1).squeeze(1)
    }
}
