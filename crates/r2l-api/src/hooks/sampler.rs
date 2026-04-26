use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor};
use r2l_core::{
    buffers::EditableTrajectoryContainer,
    env::{Env, EnvBuilder, EnvDescription},
    models::Actor,
    tensor::R2lTensor,
};
use r2l_sampler::PreprocessorY;

use crate::utils::{evaluator::Evaluator, running_mean::RunningMeanStd};

pub struct EvaluatorOptions {
    pub(crate) eval_episodes: usize,
    pub(crate) eval_freq: usize,
    pub(crate) eval_step: usize,
    pub(crate) results: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl Default for EvaluatorOptions {
    fn default() -> Self {
        Self {
            eval_episodes: 10,
            eval_freq: 10000,
            eval_step: 1000,
            results: Arc::new(Mutex::new(vec![vec![]])),
        }
    }
}

impl EvaluatorOptions {
    pub fn new(
        eval_episodes: usize,
        eval_freq: usize,
        eval_steps: usize,
    ) -> (Self, Arc<Mutex<Vec<Vec<f32>>>>) {
        let results = Arc::new(Mutex::new(vec![vec![]]));
        (
            Self {
                // device,
                eval_episodes,
                eval_freq,
                eval_step: eval_steps,
                results: results.clone(),
            },
            results,
        )
    }

    pub fn build<EB: EnvBuilder>(
        &self,
        env_builder: &EB,
        n_envs: usize,
        device: Device,
    ) -> Evaluator<EB::Env> {
        let env = env_builder.build_env().unwrap();
        Evaluator::new(
            env,
            self.eval_episodes,
            self.eval_freq * n_envs,
            self.eval_step,
            self.results.clone(),
            device,
        )
    }
}

pub struct EnvNormalizer {
    pub(crate) obs_rms: RunningMeanStd,
    pub(crate) ret_rms: RunningMeanStd,
    pub(crate) returns: Tensor,
    pub(crate) epsilon: f32,
    pub(crate) gamma: f32,
    pub(crate) clip_obs: f32,
    pub(crate) clip_rew: f32,
}

impl EnvNormalizer {
    pub fn new(
        obs_rms: RunningMeanStd,
        ret_rms: RunningMeanStd,
        returns: Tensor,
        epsilon: f32,
        gamma: f32,
        clip_obs: f32,
        clip_rew: f32,
    ) -> Self {
        Self {
            obs_rms,
            ret_rms,
            returns,
            epsilon,
            gamma,
            clip_obs,
            clip_rew,
        }
    }

    fn normalize_obs(&self, obs: Tensor) -> Result<Tensor> {
        let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
        let normalized_obs = (obs
            .broadcast_sub(&self.obs_rms.mean)?
            .broadcast_div(&self.obs_rms.var.broadcast_add(&eps)?.sqrt()?))?;
        normalized_obs.clamp(-self.clip_obs, self.clip_obs)
    }

    fn normalize_rew(&self, rew: Tensor) -> Result<Tensor> {
        let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
        let normalized_rew = (rew.broadcast_div(&self.ret_rms.var.broadcast_add(&eps)?.sqrt()?))?;
        normalized_rew.clamp(-self.clip_rew, self.clip_rew)
    }

    fn normalize_buffers<B: EditableTrajectoryContainer<Tensor = Tensor>>(
        &mut self,
        states: &mut [B],
        device: &Device,
    ) -> Result<()> {
        let n_envs = states.len();
        let obs: Vec<_> = states.iter_mut().map(|b| b.pop_last_state()).collect();
        let obs = Tensor::stack(&obs, 0)?;
        self.obs_rms.update(&obs)?;
        let obs = self.normalize_obs(obs)?;
        for (state_idx, obs) in obs.chunk(n_envs, 0)?.into_iter().enumerate() {
            let obs = obs.squeeze(0)?;
            states[state_idx].set_last_state(obs);
        }
        let rewards: Vec<_> = states.iter_mut().map(|buf| buf.pop_last_reward()).collect();
        let rewards = Tensor::from_slice(&rewards, rewards.len(), device)?;
        let gamma = Tensor::full(self.gamma, (), device)?;
        self.returns = self.returns.broadcast_mul(&gamma)?.add(&rewards)?;
        self.ret_rms.update(&self.returns)?;
        let rewards = self.normalize_rew(rewards)?;
        for (rew_idx, rew) in (rewards.to_vec1()? as Vec<f32>).iter().enumerate() {
            states[rew_idx].set_last_reward(*rew);
        }
        Ok(())
    }
}

// TODO: this needs to be reconsidered
impl<B: EditableTrajectoryContainer<Tensor = Tensor>> PreprocessorY<Tensor, B> for EnvNormalizer {
    fn preprocess_states(&mut self, _policy: &dyn Actor<Tensor = Tensor>, buffers: &mut [B]) {
        self.normalize_buffers(buffers, &Device::Cpu).unwrap();
    }
}

impl<B: EditableTrajectoryContainer<Tensor = Tensor>, E: Env<Tensor = Tensor>>
    PreprocessorY<Tensor, B> for Evaluator<E>
{
    fn preprocess_states(&mut self, policy: &dyn Actor<Tensor = Tensor>, buffers: &mut [B]) {
        let n_envs = buffers.len();
        self.evaluate(policy, n_envs).unwrap();
    }
}

pub struct NormalizerOptions {
    pub(crate) epsilon: f32,
    pub(crate) gamma: f32,
    pub(crate) clip_obs: f32,
    pub(crate) clip_rew: f32,
}

impl Default for NormalizerOptions {
    fn default() -> Self {
        Self {
            clip_obs: 10.,
            clip_rew: 10.,
            epsilon: 1e-8,
            gamma: 0.99,
        }
    }
}

impl NormalizerOptions {
    pub fn new(epsilon: f32, gamma: f32, clip_obs: f32, clip_rew: f32) -> Self {
        Self {
            epsilon,
            gamma,
            clip_obs,
            clip_rew,
        }
    }

    pub fn build<T: R2lTensor>(
        &self,
        env_description: EnvDescription<T>,
        n_envs: usize,
        device: &Device,
    ) -> EnvNormalizer {
        let obs_rms = RunningMeanStd::new(env_description.observation_size(), device.clone());
        let ret_rms = RunningMeanStd::new((), device.clone());
        let returns = Tensor::zeros(n_envs, DType::F32, device).unwrap();
        EnvNormalizer::new(
            obs_rms,
            ret_rms,
            returns,
            self.epsilon,
            self.gamma,
            self.clip_obs,
            self.clip_rew,
        )
    }
}

#[derive(Default)]
pub struct EvaluatorNormalizerOptions {
    pub(crate) evaluator_options: Option<EvaluatorOptions>,
    pub(crate) normalizer_options: Option<NormalizerOptions>,
    pub(crate) device: Option<Device>,
}

impl EvaluatorNormalizerOptions {
    pub fn evaluator(eval_options: EvaluatorOptions, device: Device) -> Self {
        Self {
            evaluator_options: Some(eval_options),
            normalizer_options: None,
            device: Some(device),
        }
    }

    pub fn normalizer(norm_options: NormalizerOptions, device: Device) -> Self {
        Self {
            evaluator_options: None,
            normalizer_options: Some(norm_options),
            device: Some(device),
        }
    }

    pub fn eval_normalizer(
        eval_options: EvaluatorOptions,
        norm_options: NormalizerOptions,
        device: Device,
    ) -> Self {
        Self {
            evaluator_options: Some(eval_options),
            normalizer_options: Some(norm_options),
            device: Some(device),
        }
    }
}

pub struct EvaluatorNormalizer<E: Env> {
    pub(crate) evaluator: Evaluator<E>,
    pub(crate) normalizer: EnvNormalizer,
    pub(crate) device: Device,
}

impl<E: Env> EvaluatorNormalizer<E> {
    pub fn new(evaluator: Evaluator<E>, normalizer: EnvNormalizer, device: Device) -> Self {
        Self {
            evaluator,
            normalizer,
            device,
        }
    }
}

impl<B: EditableTrajectoryContainer<Tensor = Tensor>, E: Env<Tensor = Tensor>>
    PreprocessorY<Tensor, B> for EvaluatorNormalizer<E>
{
    fn preprocess_states(&mut self, policy: &dyn Actor<Tensor = Tensor>, buffers: &mut [B]) {
        let n_envs = buffers.len();
        self.evaluator.evaluate(policy, n_envs).unwrap();
        self.normalizer
            .normalize_buffers(buffers, &self.device)
            .unwrap()
    }
}
