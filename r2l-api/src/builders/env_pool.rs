use crate::hooks::sequential_env_hooks::{
    EmptySequentialVecEnv, EnvNormalizer, Evaluator, EvaluatorNormalizer,
};
use candle_core::{DType, Device, Tensor};
use r2l_core::{
    env::{
        Env, EnvPoolType,
        dummy_vec_env::DummyVecEnv,
        sequential_vec_env::{SequentialVecEnv, SequentialVecEnvHooks},
    },
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};
use r2l_gym::GymEnv;

pub struct EvaluatorNormalizerOptions {
    pub epsilon: f32,
    pub gamma: f32,
    pub clip_obs: f32,
    pub clip_rew: f32,
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
}

impl Default for EvaluatorNormalizerOptions {
    fn default() -> Self {
        Self {
            clip_obs: 10.,
            clip_rew: 10.,
            epsilon: 1e-8,
            gamma: 0.99,
            eval_episodes: 5,
            eval_freq: 10000,
            eval_step: 0,
        }
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

pub enum SequentialHook<E: Env> {
    Evaluator(EvaluatorOptions),
    WithEvaluator(Evaluator<E>),
    Normalizer,
    EvaluatorNormalizer(EvaluatorNormalizerOptions),
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
                        let evaluator = Evaluator::new(
                            eval_env,
                            eval_norm_opt.eval_episodes,
                            eval_norm_opt.eval_freq * self.n_envs,
                            eval_norm_opt.eval_step,
                        );
                        let obs_rms =
                            RunningMeanStd::new(env_description.observation_size(), device.clone());
                        let ret_rms = RunningMeanStd::new((), device.clone());
                        let returns = Tensor::zeros(self.n_envs, DType::F32, device).unwrap();
                        let normalizer = EnvNormalizer::new(
                            obs_rms,
                            ret_rms,
                            returns,
                            eval_norm_opt.epsilon,
                            eval_norm_opt.gamma,
                            eval_norm_opt.clip_obs,
                            eval_norm_opt.clip_rew,
                        );
                        let hook = EvaluatorNormalizer::new(evaluator, normalizer, device.clone());
                        Box::new(hook)
                    }
                    Some(SequentialHook::Evaluator(eval_opt)) => {
                        let eval_env = GymEnv::new(&gym_env_name, None, &device).unwrap();
                        let evaluator = eval_opt.build(eval_env, self.n_envs);
                        Box::new(evaluator)
                    }
                    Some(SequentialHook::WithEvaluator(evaluator)) => Box::new(evaluator),
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
