use crate::hooks::sequential_env_hooks::{
    EmptySequentialVecEnv, EnvNormalizer, Evaluator, EvaluatorNormalizer,
};
use candle_core::{DType, Device, Tensor, cuda::cudarc::driver::result::device};
use r2l_core::{
    env::{
        Env, EnvPoolType, EnvironmentDescription,
        dummy_vec_env::DummyVecEnv,
        sequential_vec_env::{SequentialVecEnv, SequentialVecEnvHooks},
    },
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};
use r2l_gym::GymEnv;

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
    fn build(
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

pub struct EvaluatorOptions {
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
}

impl EvaluatorOptions {
    pub fn build<E: Env>(&self, eval_env: E, n_envs: usize) -> Evaluator<E> {
        Evaluator::new(
            eval_env,
            self.eval_episodes,
            self.eval_freq * n_envs,
            self.eval_step,
        )
    }
}

impl Default for EvaluatorOptions {
    fn default() -> Self {
        Self {
            eval_episodes: 10,
            eval_freq: 10000,
            eval_step: 1000,
        }
    }
}

#[derive(Default)]
pub struct EvaluatorNormalizerOptions {
    pub evaluator_options: EvaluatorOptions,
    pub normalizer_options: NormalizerOptions,
}

impl EvaluatorNormalizerOptions {
    pub fn build(
        &self,
        eval_env: GymEnv,
        n_envs: usize,
        device: Device,
    ) -> EvaluatorNormalizer<GymEnv> {
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

pub enum SequentialHook<E: Env> {
    Evaluator(EvaluatorOptions),
    WithEvaluator(Evaluator<E>),
    Normalizer,
    EvaluatorNormalizer(EvaluatorNormalizerOptions),
    WithEvaluatorNormalizer(EvaluatorNormalizer<E>),
}

pub enum VecPoolType<E: Env> {
    Dummy,
    Vec,
    Subprocessing,
    Sequential(Option<SequentialHook<E>>),
}

pub struct EnvPoolBuilder<E: Env> {
    pub pool_type: VecPoolType<E>,
    pub n_envs: usize,
    pub gym_env_name: Option<String>,
}

impl<E: Env> Default for EnvPoolBuilder<E> {
    fn default() -> Self {
        Self {
            pool_type: VecPoolType::Sequential(None),
            n_envs: 10,
            gym_env_name: None,
        }
    }
}

impl EnvPoolBuilder<GymEnv> {
    pub fn build(self, device: &Device) -> EnvPoolType<GymEnv> {
        assert!(self.n_envs >= 1, "At least env should be present");
        let Some(gym_env_name) = &self.gym_env_name else {
            todo!()
        };
        match self.pool_type {
            VecPoolType::Dummy => {
                let buffers = vec![RolloutBuffer::default(); self.n_envs];
                let env = (0..self.n_envs)
                    .map(|_| GymEnv::new(&gym_env_name, None, &device).unwrap())
                    .collect::<Vec<_>>();
                let env_description = env[0].env_description();
                EnvPoolType::Dummy(DummyVecEnv {
                    buffers,
                    env,
                    env_description,
                })
            }
            VecPoolType::Sequential(sequential_options) => {
                let buffers = vec![RolloutBuffer::default(); self.n_envs];
                let envs = (0..self.n_envs)
                    .map(|_| GymEnv::new(&gym_env_name, None, &device).unwrap())
                    .collect::<Vec<_>>();
                let env_description = envs[0].env_description();
                let hooks: Box<dyn SequentialVecEnvHooks> = match sequential_options {
                    None => Box::new(EmptySequentialVecEnv),
                    Some(SequentialHook::EvaluatorNormalizer(eval_norm_opt)) => {
                        let eval_env = GymEnv::new(&gym_env_name, None, &device).unwrap();
                        let hook = eval_norm_opt.build(eval_env, self.n_envs, device.clone());
                        Box::new(hook)
                    }
                    Some(SequentialHook::Evaluator(eval_opt)) => {
                        let eval_env = GymEnv::new(&gym_env_name, None, &device).unwrap();
                        let evaluator = eval_opt.build(eval_env, self.n_envs);
                        Box::new(evaluator)
                    }
                    Some(SequentialHook::WithEvaluator(evaluator)) => Box::new(evaluator),
                    Some(SequentialHook::WithEvaluatorNormalizer(eval_normalizer)) => {
                        Box::new(eval_normalizer)
                    }
                    _ => todo!(),
                };
                EnvPoolType::Sequential(SequentialVecEnv {
                    buffers,
                    envs,
                    env_description,
                    hooks,
                })
            }
            _ => todo!(),
        }
    }
}
