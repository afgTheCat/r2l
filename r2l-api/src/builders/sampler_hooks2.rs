use crate::{
    builders::env::EnvBuilderTrait,
    hooks::sampler::{EnvNormalizer, EvaluatorNormalizer},
    utils::{evaluator::Evaluator, running_mean::RunningMeanStd},
};
use candle_core::{DType, Device, Tensor};
use r2l_buffer::Buffer;
use r2l_core::{
    env::{Env, EnvironmentDescription},
    sampler::SequntialStepBoundHooks,
};
use std::sync::{Arc, Mutex};

pub struct EvaluatorOptions {
    pub device: Device,
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
    pub results: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl Default for EvaluatorOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            eval_episodes: 10,
            eval_freq: 10000,
            eval_step: 1000,
            results: Arc::new(Mutex::new(vec![vec![]])),
        }
    }
}

impl EvaluatorOptions {
    pub fn new(
        device: Device,
        eval_episodes: usize,
        eval_freq: usize,
        eval_steps: usize,
    ) -> (Self, Arc<Mutex<Vec<Vec<f32>>>>) {
        let results = Arc::new(Mutex::new(vec![vec![]]));
        (
            Self {
                device,
                eval_episodes,
                eval_freq,
                eval_step: eval_steps,
                results: results.clone(),
            },
            results,
        )
    }

    pub fn build<EB: EnvBuilderTrait>(
        &self,
        env_builder: &EB,
        n_envs: usize,
    ) -> Evaluator<EB::Env> {
        let env = env_builder.build_env(&self.device).unwrap();
        Evaluator::new(
            env,
            self.eval_episodes,
            self.eval_freq * n_envs,
            self.eval_step,
            self.results.clone(),
        )
    }
}

pub struct NormalizerOptions {
    pub epsilon: f32,
    pub gamma: f32,
    pub clip_obs: f32,
    pub clip_rew: f32,
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

    pub fn build<T>(
        &self,
        env_description: EnvironmentDescription<T>,
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
    pub evaluator_options: Option<EvaluatorOptions>,
    pub normalizer_options: Option<NormalizerOptions>,
}

impl EvaluatorNormalizerOptions {
    pub fn evaluator(eval_options: EvaluatorOptions) -> Self {
        Self {
            evaluator_options: Some(eval_options),
            normalizer_options: None,
        }
    }

    pub fn normalizer(norm_options: NormalizerOptions) -> Self {
        Self {
            evaluator_options: None,
            normalizer_options: Some(norm_options),
        }
    }

    pub fn eval_normalizer(
        eval_options: EvaluatorOptions,
        norm_options: NormalizerOptions,
    ) -> Self {
        Self {
            evaluator_options: Some(eval_options),
            normalizer_options: Some(norm_options),
        }
    }
}

impl EvaluatorNormalizerOptions {
    pub fn build<E: Env<Tensor = Buffer> + 'static, EB: EnvBuilderTrait<Env = E>>(
        &self,
        env_description: EnvironmentDescription<<EB::Env as Env>::Tensor>,
        env_builder: &EB,
        n_envs: usize,
        device: &Device,
    ) -> Option<Box<dyn SequntialStepBoundHooks<EB::Env>>> {
        match &self {
            EvaluatorNormalizerOptions {
                evaluator_options: None,
                normalizer_options: None,
            } => None,
            EvaluatorNormalizerOptions {
                evaluator_options: Some(eval_options),
                normalizer_options: None,
            } => {
                let evaluator = eval_options.build(env_builder, n_envs);
                Some(Box::new(evaluator))
            }
            EvaluatorNormalizerOptions {
                evaluator_options: None,
                normalizer_options: Some(norm_options),
            } => {
                let normalizer = norm_options.build(env_description, n_envs, device);
                Some(Box::new(normalizer))
            }
            EvaluatorNormalizerOptions {
                evaluator_options: Some(eval_options),
                normalizer_options: Some(norm_options),
            } => {
                let evaluator = eval_options.build(env_builder, n_envs);
                let normalizer = norm_options.build(env_description, n_envs, device);
                Some(Box::new(EvaluatorNormalizer {
                    evaluator,
                    normalizer,
                    device: device.clone(),
                }))
            }
        }
    }
}
