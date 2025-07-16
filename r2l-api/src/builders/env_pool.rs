use crate::hooks::sequential_env_hooks::{
    EmptySequentialVecEnv, EnvNormalizer, Evaluator, EvaluatorNormalizer,
};
use candle_core::{DType, Device, Result, Tensor};
use crossbeam::sync::ShardedLock;
use r2l_core::{
    distributions::Distribution,
    env::{
        Env, EnvPoolType, EnvironmentDescription,
        dummy_vec_env::DummyVecEnv,
        sequential_vec_env::{SequentialVecEnv, SequentialVecEnvHooks},
        vec_env::{VecEnv, WorkerTask, WorkerThread},
    },
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};
use r2l_gym::GymEnv;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

pub trait EnvBuilderTrait: Sync {
    type Env: Env;

    fn build_env(&self) -> Result<Self::Env>;
}

impl EnvBuilderTrait for String {
    type Env = GymEnv;

    fn build_env(&self) -> Result<Self::Env> {
        GymEnv::new(&self, None, &Device::Cpu)
    }
}

impl<E: Env, F: Sync> EnvBuilderTrait for F
where
    F: Fn() -> Result<E>,
{
    type Env = E;

    fn build_env(&self) -> Result<Self::Env> {
        (self)()
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

pub enum VecPoolType {
    Dummy,
    Vec,
    Subprocessing,
    Sequential(SequentialEnvHookTypes),
}

// TODO: we should bundle the builders for the hooks here as well
pub enum SequentialEnvHookTypes {
    None,
    EvaluatorOnly { options: EvaluatorOptions },
    NormalizerOnly { options: NormalizerOptions },
    EvaluatorNormalizer { options: EvaluatorNormalizerOptions },
}

pub struct EnvPoolBuilder {
    pub pool_type: VecPoolType,
    pub n_envs: usize,
}

impl Default for EnvPoolBuilder {
    fn default() -> Self {
        Self {
            pool_type: VecPoolType::Sequential(SequentialEnvHookTypes::None),
            n_envs: 10,
        }
    }
}

struct BuildersWithIdx<EB: EnvBuilderTrait> {
    builders: Vec<EB>,
    idx: AtomicUsize,
}

enum BuilderType<EB: EnvBuilderTrait> {
    EnvBuilder(EB),
    EnvBuilderVec(BuildersWithIdx<EB>),
}

impl<EB: EnvBuilderTrait> BuilderType<EB> {
    fn env_buillder_vec(env_builders: Vec<EB>) -> Self {
        Self::EnvBuilderVec(BuildersWithIdx {
            builders: env_builders,
            idx: AtomicUsize::new(0),
        })
    }

    fn build_all_envs_and_buffers(
        &self,
        n_envs: usize,
    ) -> Result<(Vec<RolloutBuffer>, Vec<EB::Env>)> {
        match self {
            Self::EnvBuilder(eb) => {
                let buffers = vec![RolloutBuffer::default(); n_envs];
                let envs = (0..n_envs)
                    .map(|_| eb.build_env())
                    .collect::<Result<Vec<_>>>()?;
                Ok((buffers, envs))
            }
            Self::EnvBuilderVec(BuildersWithIdx { builders, .. }) => {
                let buffers = vec![RolloutBuffer::default(); builders.len()];
                let envs = builders
                    .iter()
                    .map(|ebs| ebs.build_env())
                    .collect::<Result<Vec<_>>>()?;
                Ok((buffers, envs))
            }
        }
    }

    fn build_single_env(&self) -> Result<EB::Env> {
        match self {
            Self::EnvBuilder(eb) => eb.build_env(),
            Self::EnvBuilderVec(BuildersWithIdx { builders, idx }) => {
                let i = idx.fetch_add(1, Ordering::Relaxed);
                builders[i % builders.len()].build_env()
            }
        }
    }
}

impl EnvPoolBuilder {
    pub fn set_env_pool_type(&mut self, pool_type: VecPoolType) {
        self.pool_type = pool_type
    }

    fn build_inner<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builder: BuilderType<EB>,
    ) -> Result<EnvPoolType<E>> {
        match self.pool_type {
            VecPoolType::Dummy => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers(self.n_envs)?;
                let env_description = envs[0].env_description();
                Ok(EnvPoolType::Dummy(DummyVecEnv {
                    buffers,
                    env: envs,
                    env_description,
                }))
            }
            VecPoolType::Sequential(hook_types) => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers(self.n_envs)?;
                let env_description = envs[0].env_description();
                let hooks: Box<dyn SequentialVecEnvHooks> = match hook_types {
                    SequentialEnvHookTypes::None => Box::new(EmptySequentialVecEnv),
                    // TODO: this should also be a hook
                    // SequentialEnvHookTypes::NormalizerOnly { options } => {
                    //     let normalizer = options.build(env_description, self.n_envs, device);
                    //     Box::new(normalizer)
                    // }
                    SequentialEnvHookTypes::EvaluatorOnly { options } => {
                        let eval_env = env_builder.build_single_env()?;
                        let evaluator = options.build(eval_env, self.n_envs);
                        Box::new(evaluator)
                    }
                    SequentialEnvHookTypes::EvaluatorNormalizer { options } => {
                        let eval_env = env_builder.build_single_env()?;
                        let eval_normalizer = options.build(eval_env, self.n_envs, device.clone());
                        Box::new(eval_normalizer)
                    }
                    _ => todo!(),
                };
                Ok(EnvPoolType::Sequential(SequentialVecEnv {
                    buffers,
                    envs,
                    env_description,
                    hooks,
                }))
            }
            VecPoolType::Vec => {
                let (result_tx, result_rx) = crossbeam::channel::unbounded::<RolloutBuffer>();
                let mut worker_txs = vec![];
                let distr_lock = Arc::new(ShardedLock::new(None::<&'static dyn Distribution>));
                let task_rxs = (0..self.n_envs)
                    .map(|_| {
                        let (task_tx, task_rx) = crossbeam::channel::unbounded::<WorkerTask>();
                        worker_txs.push(task_tx);
                        task_rx
                    })
                    .collect::<Vec<_>>();
                // TODO: this is kinda stupid that we need to extract this from a thread
                let env_description = Mutex::new(None);
                crossbeam::thread::scope(|scope| {
                    for task_rx in task_rxs {
                        let distr_lock = distr_lock.clone();
                        scope.spawn(|_| {
                            let buff = RolloutBuffer::default();
                            let env = env_builder
                                .build_single_env()
                                .expect("Could not build environment");
                            let mut env_description = env_description.lock().unwrap();
                            env_description.replace(env.env_description());
                            let mut worker = WorkerThread {
                                env,
                                buff,
                                task_rx,
                                result_tx: result_tx.clone(),
                            };
                            worker.work(distr_lock);
                        });
                    }
                })
                .expect("Could not spawn worker threads");
                let env_description = env_description.lock().unwrap();
                let vec_env = VecEnv {
                    worker_txs,
                    result_rx,
                    env_description: env_description.as_ref().unwrap().clone(),
                    distr_lock,
                };
                Ok(EnvPoolType::VecEnv(vec_env))
            }
            _ => todo!(),
        }
    }

    // TODO: new name
    pub fn build2<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builders: Vec<EB>,
    ) -> Result<EnvPoolType<E>> {
        self.build_inner(device, BuilderType::env_buillder_vec(env_builders))
    }

    // TODO: new name
    pub fn build<E: Env + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builder: EB,
    ) -> Result<EnvPoolType<E>> {
        self.build_inner(device, BuilderType::EnvBuilder(env_builder))
    }
}
