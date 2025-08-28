use crate::{
    hooks::sampler::{EnvNormalizer, EvaluatorNormalizer},
    utils::{evaluator::Evaluator, running_mean::RunningMeanStd},
};
use candle_core::{DType, Device, Tensor};
use r2l_core::{
    env::{Env, EnvironmentDescription},
    numeric::Buffer,
    sampler::samplers::SequntialStepBoundHooks,
};
use std::sync::{Arc, Mutex};

pub struct EvaluatorOptions {
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
    pub results: Arc<Mutex<Vec<Vec<f32>>>>,
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
                eval_episodes: eval_episodes,
                eval_freq: eval_freq,
                eval_step: eval_steps,
                results: results.clone(),
            },
            results,
        )
    }

    pub fn build<E: Env<Tensor = Buffer>>(&self, eval_env: E, n_envs: usize) -> Evaluator<E> {
        Evaluator::new(
            eval_env,
            self.eval_episodes,
            self.eval_freq * n_envs,
            self.eval_step,
            self.results.clone(),
        )
    }
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

    pub fn build(
        &self,
        env_description: EnvironmentDescription,
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
    pub evaluator_options: EvaluatorOptions,
    pub normalizer_options: NormalizerOptions,
}

impl EvaluatorNormalizerOptions {
    pub fn new(evaluator_options: EvaluatorOptions, normalizer_options: NormalizerOptions) -> Self {
        Self {
            evaluator_options,
            normalizer_options,
        }
    }

    pub fn build<E: Env<Tensor = Buffer>>(
        &self,
        eval_env: E,
        n_envs: usize,
        device: Device,
    ) -> EvaluatorNormalizer<E> {
        let env_description = eval_env.env_description();
        let normalizer = self
            .normalizer_options
            .build(env_description, n_envs, &device);
        let evaluator = self.evaluator_options.build(eval_env, n_envs);
        EvaluatorNormalizer {
            evaluator,
            normalizer,
            device,
        }
    }
}

#[derive(Default)]
pub enum SequentialEnvHookTypes {
    #[default]
    None,
    EvaluatorOnly {
        options: EvaluatorOptions,
    },
    NormalizerOnly {
        options: NormalizerOptions,
    },
    EvaluatorNormalizer {
        options: EvaluatorNormalizerOptions,
    },
}

impl SequentialEnvHookTypes {
    pub fn build<E: Env>(&self) -> Option<Box<dyn SequntialStepBoundHooks<E>>> {
        match &self {
            Self::None => None,
            Self::EvaluatorOnly { options } => {
                // let evaluator = options.build(eval_env, n_envs)
                todo!()
            }
            Self::NormalizerOnly { options } => {
                todo!()
            }
            Self::EvaluatorNormalizer { options } => {
                todo!()
            }
        }
    }
}
