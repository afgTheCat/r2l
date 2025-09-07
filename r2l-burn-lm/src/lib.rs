// TODO: burn support is highly experimental, and we need to figure out the right abstractions at
// one point. Maybe next release.
pub mod burn_rollout_buffer;
mod distributions;
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

// fn train_single_batch<B: AutodiffBackend>(
//     ppo_debug: &mut BurnPPODebug<B>,
//     states: Tensor<B, 2>,
//     returns: Tensor<B, 1>,
//     value_losses: &mut Vec<f32>,
//     logp_old: Tensor<B, 1>,
//     advantages: Tensor<B, 1>,
//     actions: Tensor<B, 2>,
// ) {
//     let logps = ppo_debug.logp(states.clone(), actions);
//     let logp_diff = logps - logp_old;
//     let ratio = logp_diff.exp();
//     let values_pred = ppo_debug.model.value_net.forward(states).squeeze(1);
//     let value_min_val_pred = returns - values_pred;
//     let value_loss = (value_min_val_pred.clone() * value_min_val_pred).mean();
//     value_losses.push(value_loss.clone().into_scalar().to_f32());
//     let clip_adv = ratio.clone().clamp(1. - CLIP_RANGE, 1. + CLIP_RANGE) * advantages.clone();
//     let policy_loss = (-(ratio * advantages).min_pair(clip_adv)).mean();
//     let loss: Tensor<B, 1> = value_loss + policy_loss;
//     let grads = loss.backward();
//     let grads = GradientsParams::from_grads(grads, &ppo_debug.model);
//     ppo_debug.model = ppo_debug
//         .optimizer
//         .step(1e-4, ppo_debug.model.clone(), grads);
// }
//
// fn minibatch_learning<B: AutodiffBackend>(
//     ppo_debug: &mut BurnPPODebug<B>,
//     states: Tensor<B, 2>,
//     returns: Tensor<B, 1>,
//     indicies: &[i64],
//     value_losses: &mut Vec<f32>,
//     logp_old: Tensor<B, 1>,
//     advantages: Tensor<B, 1>,
//     actions: Tensor<B, 2>,
// ) {
//     let device = Default::default();
//     let mut steps_processed = 0;
//     while steps_processed < N_STEPS && steps_processed + BATCH_SIZE <= N_STEPS {
//         let indices = &indicies[steps_processed..steps_processed + BATCH_SIZE];
//         let indices: Tensor<B, 1, Int> = Tensor::from_data(indices, &device);
//         let states = states.clone().select(0, indices.clone());
//         let returns = returns.clone().select(0, indices.clone());
//         let advantages = advantages.clone().select(0, indices.clone());
//         let actions = actions.clone().select(0, indices.clone());
//         let logp_old = logp_old.clone().select(0, indices.clone());
//         train_single_batch(
//             ppo_debug,
//             states,
//             returns,
//             value_losses,
//             logp_old,
//             advantages,
//             actions,
//         );
//         steps_processed += BATCH_SIZE;
//     }
// }

// // learning for one epoch
// fn epoch_loop<B: AutodiffBackend>(
//     burn_ppo: &mut BurnPPODebug<B>,
//     states: Tensor<B, 2>,
//     returns: Tensor<B, 1>,
//     value_losses: &mut Vec<f32>,
//     logp_old: Tensor<B, 1>,
//     advantages: Tensor<B, 1>,
//     actions: Tensor<B, 2>,
// ) {
//     let mut indicies = (0..N_STEPS).map(|i| i as i64).collect::<Vec<_>>();
//     indicies.shuffle(&mut rand::rng());
//     minibatch_learning(
//         burn_ppo,
//         states.clone(),
//         returns.clone(),
//         &indicies,
//         value_losses,
//         logp_old.clone(),
//         advantages.clone(),
//         actions.clone(),
//     );
// }
//
// fn train<B: AutodiffBackend>(burn_ppo: &mut BurnPPODebug<B>) {
//     // Loop is the sameish
//     for _ in 0..TOTAL_ROLLOUTS {
//         let infer_model = burn_ppo.model.valid();
//         let (states, returns, logp_old, advantages, actions) =
//             collect_rollouts::<B>(infer_model, &burn_ppo.env);
//         let mut value_losses = vec![];
//         for _ in 0..10 {
//             epoch_loop(
//                 burn_ppo,
//                 states.clone(),
//                 returns.clone(),
//                 &mut value_losses,
//                 logp_old.clone(),
//                 advantages.clone(),
//                 actions.clone(),
//             );
//         }
//     }
// }
//
// fn build_ppo<B: AutodiffBackend>() -> BurnPPODebug<B> {
//     let device = Default::default();
//     let (env, action_size, obs_size, low, high) = build_env();
//     let low_data = TensorData::new(low, Shape::new([1, action_size]));
//     let low: Tensor<B, 2> = Tensor::from_data(low_data, &device);
//     let high_data = TensorData::new(high, Shape::new([1, action_size]));
//     let high: Tensor<B, 2> = Tensor::from_data(high_data, &device);
//     let value_net: Sequential<B> = Sequential::build(&[obs_size, 64, 64, 1]);
//     let mu_net: Sequential<B> = Sequential::build(&[obs_size, 64, 64, action_size]);
//     let log_std: Tensor<B, 2> = Tensor::random(
//         Shape::new([1, obs_size]),
//         burn::tensor::Distribution::Normal(0., 1.),
//         &device,
//     );
//     let model = Model {
//         mu_net,
//         value_net,
//         log_std,
//         obs_size,
//         action_low: low.clone(),
//         action_high: high.clone(),
//     };
//     let optimizer: OptimizerAdaptor<AdamW, Model<B>, B> = AdamWConfig::new().init();
//     BurnPPODebug { model, optimizer }
// }
//
// pub fn burn_ppo_loop<B: AutodiffBackend>() {
//     let mut ppo = build_ppo::<B>();
//     train(&mut ppo);
// }
