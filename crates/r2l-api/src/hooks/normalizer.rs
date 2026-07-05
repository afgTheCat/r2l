// use candle_core::{DType, Device, Result, Tensor};
// use r2l_core::{env::EnvDescription, tensor::R2lTensor};

// use crate::utils::running_mean::RunningMeanStd;

// pub struct EnvNormalizer {
//     pub(crate) obs_rms: RunningMeanStd,
//     pub(crate) ret_rms: RunningMeanStd,
//     pub(crate) returns: Tensor,
//     pub(crate) epsilon: f32,
//     pub(crate) gamma: f32,
//     pub(crate) clip_obs: f32,
//     pub(crate) clip_rew: f32,
// }

// impl EnvNormalizer {
//     pub fn new(
//         obs_rms: RunningMeanStd,
//         ret_rms: RunningMeanStd,
//         returns: Tensor,
//         epsilon: f32,
//         gamma: f32,
//         clip_obs: f32,
//         clip_rew: f32,
//     ) -> Self {
//         Self {
//             obs_rms,
//             ret_rms,
//             returns,
//             epsilon,
//             gamma,
//             clip_obs,
//             clip_rew,
//         }
//     }
//
//     fn normalize_obs(&self, obs: Tensor) -> Result<Tensor> {
//         let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
//         let normalized_obs = (obs
//             .broadcast_sub(&self.obs_rms.mean)?
//             .broadcast_div(&self.obs_rms.var.broadcast_add(&eps)?.sqrt()?))?;
//         normalized_obs.clamp(-self.clip_obs, self.clip_obs)
//     }
//
//     fn normalize_rew(&self, rew: Tensor) -> Result<Tensor> {
//         let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
//         let normalized_rew = (rew.broadcast_div(&self.ret_rms.var.broadcast_add(&eps)?.sqrt()?))?;
//         normalized_rew.clamp(-self.clip_rew, self.clip_rew)
//     }
//
//     fn normalize_buffers<B: EditableTrajectoryContainer<Tensor = Tensor>>(
//         &mut self,
//         states: &mut [B],
//         device: &Device,
//     ) -> Result<()> {
//         let n_envs = states.len();
//         let obs: Vec<_> = states.iter_mut().map(|b| b.pop_last_state()).collect();
//         let obs = Tensor::stack(&obs, 0)?;
//         self.obs_rms.update(&obs)?;
//         let obs = self.normalize_obs(obs)?;
//         for (state_idx, obs) in obs.chunk(n_envs, 0)?.into_iter().enumerate() {
//             let obs = obs.squeeze(0)?;
//             states[state_idx].set_last_state(obs);
//         }
//         let rewards: Vec<_> = states.iter_mut().map(|buf| buf.pop_last_reward()).collect();
//         let rewards = Tensor::from_slice(&rewards, rewards.len(), device)?;
//         let gamma = Tensor::full(self.gamma, (), device)?;
//         self.returns = self.returns.broadcast_mul(&gamma)?.add(&rewards)?;
//         self.ret_rms.update(&self.returns)?;
//         let rewards = self.normalize_rew(rewards)?;
//         for (rew_idx, rew) in (rewards.to_vec1()? as Vec<f32>).iter().enumerate() {
//             states[rew_idx].set_last_reward(*rew);
//         }
//         Ok(())
//     }
//
//     // NOTE: old hook impl
//     fn preprocess_states<B: EditableTrajectoryContainer<Tensor = Tensor>>(
//         &mut self,
//         _policy: &dyn Actor<Tensor = Tensor>,
//         buffers: &mut [B],
//     ) {
//         self.normalize_buffers(buffers, &Device::Cpu).unwrap();
//     }
// }

// pub struct NormalizerOptions {
//     pub(crate) epsilon: f32,
//     pub(crate) gamma: f32,
//     pub(crate) clip_obs: f32,
//     pub(crate) clip_rew: f32,
// }

// impl Default for NormalizerOptions {
//     fn default() -> Self {
//         Self {
//             clip_obs: 10.,
//             clip_rew: 10.,
//             epsilon: 1e-8,
//             gamma: 0.99,
//         }
//     }
// }

// impl NormalizerOptions {
//     pub fn new(epsilon: f32, gamma: f32, clip_obs: f32, clip_rew: f32) -> Self {
//         Self {
//             epsilon,
//             gamma,
//             clip_obs,
//             clip_rew,
//         }
//     }
//
//     pub fn build<T: R2lTensor>(
//         &self,
//         env_description: EnvDescription<T>,
//         n_envs: usize,
//         device: &Device,
//     ) -> EnvNormalizer {
//         let obs_rms = RunningMeanStd::new(env_description.observation_size(), device.clone());
//         let ret_rms = RunningMeanStd::new((), device.clone());
//         let returns = Tensor::zeros(n_envs, DType::F32, device).unwrap();
//         EnvNormalizer::new(
//             obs_rms,
//             ret_rms,
//             returns,
//             self.epsilon,
//             self.gamma,
//             self.clip_obs,
//             self.clip_rew,
//         )
//     }
// }

use r2l_core::tensor::R2lTensor;
use r2l_sampler::ClippedNormalizer;

struct EvaluatorWithNormalizer<T: R2lTensor> {
    normalizer: ClippedNormalizer<T>,
}
