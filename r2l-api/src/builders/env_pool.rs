use crate::{
    hooks::env_pool::{EmptySequentialVecEnv, EnvNormalizer, EvaluatorNormalizer},
    utils::{evaluator::Evaluator, running_mean::RunningMeanStd},
};
use candle_core::{DType, Device, Result, Tensor};
use crossbeam::sync::ShardedLock;
use r2l_core::{
    distributions::Distribution,
    env::{Env, EnvironmentDescription},
    env_pools::{
        R2lEnvHolder, R2lEnvPool, SequentialVecEnvHooks, StepMode,
        thread_env_holder::{ThreadResult, WorkerTask, WorkerThread},
        vector_env_holder::VecEnvHolder,
    },
    utils::rollout_buffer::RolloutBuffer,
};
use r2l_gym::GymEnv;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

pub trait EnvBuilderTrait: Sync {
    type Env: Env;

    fn build_env(&self, device: &Device) -> Result<Self::Env>;
}

impl EnvBuilderTrait for String {
    type Env = GymEnv;

    fn build_env(&self, device: &Device) -> Result<Self::Env> {
        GymEnv::new(&self, None, device)
    }
}

impl<E: Env, F: Sync> EnvBuilderTrait for F
where
    F: Fn(&Device) -> Result<E>,
{
    type Env = E;

    fn build_env(&self, device: &Device) -> Result<Self::Env> {
        (self)(device)
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

    pub fn build<E: Env>(&self, eval_env: E, n_envs: usize) -> Evaluator<E> {
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

    pub fn build<E: Env>(
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

pub enum VecPoolType {
    Dummy,
    Vec,
    Subprocessing,
    Sequential(SequentialEnvHookTypes),
}

impl Default for VecPoolType {
    fn default() -> Self {
        VecPoolType::Sequential(SequentialEnvHookTypes::default())
    }
}

pub enum BuilderType<EB: EnvBuilderTrait> {
    EnvBuilder { builder: EB, n_envs: usize },
    EnvBuilderVec { builders: Vec<EB>, idx: AtomicUsize },
}

impl<EB: EnvBuilderTrait> BuilderType<EB> {
    pub fn env_buillder_vec(env_builders: Vec<EB>) -> Self {
        Self::EnvBuilderVec {
            builders: env_builders,
            idx: AtomicUsize::new(0),
        }
    }

    pub fn env_builder(builder: EB, n_envs: usize) -> Self {
        Self::EnvBuilder { builder, n_envs }
    }

    pub fn build_all_envs_and_buffers(
        &self,
        device: &Device,
    ) -> Result<(Vec<RolloutBuffer>, Vec<EB::Env>)> {
        match self {
            Self::EnvBuilder { builder, n_envs } => {
                let buffers = vec![RolloutBuffer::default(); *n_envs];
                let envs = (0..*n_envs)
                    .map(|_| builder.build_env(device))
                    .collect::<Result<Vec<_>>>()?;
                Ok((buffers, envs))
            }
            Self::EnvBuilderVec { builders, .. } => {
                let n_envs = builders.len();
                let buffers = vec![RolloutBuffer::default(); n_envs];
                let envs = builders
                    .iter()
                    .map(|ebs| ebs.build_env(device))
                    .collect::<Result<Vec<_>>>()?;
                Ok((buffers, envs))
            }
        }
    }

    pub fn build_single_env(&self, device: &Device) -> Result<EB::Env> {
        match self {
            Self::EnvBuilder { builder, .. } => builder.build_env(device),
            Self::EnvBuilderVec { builders, idx } => {
                let i = idx.fetch_add(1, Ordering::Relaxed);
                builders[i % builders.len()].build_env(device)
            }
        }
    }

    pub fn n_envs(&self) -> usize {
        match self {
            Self::EnvBuilder { n_envs, .. } => *n_envs,
            Self::EnvBuilderVec { builders, .. } => builders.len(),
        }
    }

    pub fn build_all_envs(&self) -> Vec<RolloutBuffer> {
        match self {
            Self::EnvBuilder { n_envs, .. } => vec![RolloutBuffer::default(); *n_envs],
            Self::EnvBuilderVec { builders, .. } => vec![RolloutBuffer::default(); builders.len()],
        }
    }
}

impl VecPoolType {
    pub fn to_r2l_pool_inner<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        &self,
        device: &Device,
        env_builder: BuilderType<EB>,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        match self {
            Self::Dummy => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers(device)?;
                let env_description = envs[0].env_description();
                Ok(R2lEnvPool {
                    env_holder: R2lEnvHolder::Vec(VecEnvHolder { envs, buffers }),
                    step_mode: StepMode::Async,
                    env_description,
                })
            }
            Self::Sequential(hook_types) => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers(device)?;
                let env_description = envs[0].env_description();
                let hooks: Box<dyn SequentialVecEnvHooks> = match hook_types {
                    SequentialEnvHookTypes::None => Box::new(EmptySequentialVecEnv),
                    // TODO: this should also be a hook
                    // SequentialEnvHookTypes::NormalizerOnly { options } => {
                    //     let normalizer = options.build(env_description, self.n_envs, device);
                    //     Box::new(normalizer)
                    // }
                    SequentialEnvHookTypes::EvaluatorOnly { options } => {
                        let eval_env = env_builder.build_single_env(device)?;
                        let n_envs = env_builder.n_envs();
                        let evaluator = options.build(eval_env, n_envs);
                        Box::new(evaluator)
                    }
                    SequentialEnvHookTypes::EvaluatorNormalizer { options } => {
                        let eval_env = env_builder.build_single_env(device)?;
                        let n_envs = env_builder.n_envs();
                        let eval_normalizer = options.build(eval_env, n_envs, device.clone());
                        Box::new(eval_normalizer)
                    }
                    _ => todo!(),
                };
                Ok(R2lEnvPool {
                    env_holder: R2lEnvHolder::Vec(VecEnvHolder { envs, buffers }),
                    step_mode: StepMode::Sequential(hooks),
                    env_description,
                })
            }
            Self::Vec => {
                let (result_tx, result_rx) = crossbeam::channel::unbounded::<ThreadResult>();
                let mut worker_txs = vec![];
                let distr_lock = Arc::new(ShardedLock::new(None::<&'static dyn Distribution>));
                let n_envs = env_builder.n_envs();
                let task_rxs = (0..n_envs)
                    .map(|_| {
                        let (task_tx, task_rx) = crossbeam::channel::unbounded::<WorkerTask>();
                        worker_txs.push(task_tx);
                        task_rx
                    })
                    .collect::<Vec<_>>();
                // TODO: this is kinda stupid that we need to extract this from a thread
                // Maybe it would make sense to have the environment builder implement a get env
                // description function also. Usually the env descriptions are only needed during
                // the envrioments build time. For now it's ok, as this makes the simplest API, but
                // we probably want to revisit this problem
                let env_description = Mutex::new(None);
                crossbeam::thread::scope(|scope| {
                    for task_rx in task_rxs {
                        let distr_lock = distr_lock.clone();
                        scope.spawn(|_| {
                            let env = env_builder
                                .build_single_env(device)
                                .expect("Could not build environment");
                            let mut env_description = env_description.lock().unwrap();
                            env_description.replace(env.env_description());
                            let mut worker = WorkerThread {
                                env,
                                task_rx,
                                result_tx: result_tx.clone(),
                            };
                            worker.work(distr_lock);
                        });
                    }
                })
                .expect("Could not spawn worker threads");
                let env_description = env_description.lock().unwrap();
                let env_description = env_description.as_ref().unwrap();
                let buffs = env_builder.build_all_envs();
                // let env_pool = R2lEnvPool {
                //     env_holder: R2lEnvHolder::Thread(ThreadHolder {
                //         current_states: None,
                //         worker_txs,
                //         result_rx,
                //         distr_lock,
                //         buffs,
                //     }),
                //     step_mode: StepMode::Async,
                //     env_description,
                // };
                todo!()
            }
            _ => todo!(),
        }
    }

    pub fn build_with_builders<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builders: Vec<EB>,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        self.to_r2l_pool_inner(device, BuilderType::env_buillder_vec(env_builders))
    }

    pub fn build<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builder: EB,
        n_envs: usize,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        self.to_r2l_pool_inner(device, BuilderType::env_builder(env_builder, n_envs))
    }
}
