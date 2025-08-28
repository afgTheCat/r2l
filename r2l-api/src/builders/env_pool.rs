use crate::{
    builders::{env::EnvBuilderTrait, sampler_hooks::SequentialEnvHookTypes},
    hooks::sampler::EmptySequentialVecEnv,
};
use candle_core::{Device, Result, Tensor};
use crossbeam::sync::ShardedLock;
use r2l_core::{
    distributions::Distribution,
    env::{Env, RolloutMode},
    env_pools::{
        R2lEnvHolder, R2lEnvPool, SequentialVecEnvHooks, StepMode,
        thread_env_holder::{ThreadResult, WorkerTask, WorkerThread},
        vector_env_holder::VecEnvHolder,
    },
    numeric::Buffer,
    sampler::samplers::step_bound_sampler::VecEnvHolder2,
    utils::rollout_buffer::RolloutBuffer,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

pub enum VecPoolType {
    Dummy,
    Dummy2,
    Vec,
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

    // TODO: maybe build the combined buffer here?
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
    pub fn to_r2l_pool_inner<E: Env<Tensor = Buffer> + 'static, EB: EnvBuilderTrait<Env = E>>(
        &self,
        device: &Device,
        env_builder: BuilderType<EB>,
        rollout_mode: RolloutMode,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        match self {
            Self::Dummy => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers(device)?;
                let env_description = envs[0].env_description();
                Ok(R2lEnvPool {
                    env_holder: R2lEnvHolder::Vec(VecEnvHolder {
                        envs,
                        buffers,
                        device: device.clone(),
                    }),
                    step_mode: StepMode::Async,
                    env_description,
                    rollout_mode,
                })
            }
            Self::Dummy2 => {
                let (buffers, envs) = env_builder.build_all_envs_and_buffers(device)?;
                let env_description = envs[0].env_description();
                let vec_env_holder = match rollout_mode {
                    RolloutMode::StepBound { n_steps } => VecEnvHolder2::new(n_steps, envs),
                    RolloutMode::EpisodeBound { n_episodes } => todo!(),
                };
                Ok(R2lEnvPool {
                    env_holder: R2lEnvHolder::Vec2(vec_env_holder),
                    step_mode: StepMode::Async,
                    env_description,
                    rollout_mode,
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
                    env_holder: R2lEnvHolder::Vec(VecEnvHolder {
                        envs,
                        buffers,
                        device: device.clone(),
                    }),
                    step_mode: StepMode::Sequential(hooks),
                    env_description,
                    rollout_mode,
                })
            }
            Self::Vec => {
                let (result_tx, result_rx) = crossbeam::channel::unbounded::<ThreadResult>();
                let mut worker_txs = vec![];
                let distr_lock = Arc::new(ShardedLock::new(
                    None::<&'static dyn Distribution<Tensor = Tensor>>,
                ));
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
                                device: device.clone(),
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
        }
    }

    pub fn build_with_builders<E: Env<Tensor = Buffer> + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builders: Vec<EB>,
        rollout_mode: RolloutMode,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        self.to_r2l_pool_inner(
            device,
            BuilderType::env_buillder_vec(env_builders),
            rollout_mode,
        )
    }

    pub fn build<E: Env<Tensor = Buffer> + 'static, EB: EnvBuilderTrait<Env = E>>(
        self,
        device: &Device,
        env_builder: EB,
        n_envs: usize,
        rollout_mode: RolloutMode,
    ) -> Result<R2lEnvPool<R2lEnvHolder<E>>> {
        self.to_r2l_pool_inner(
            device,
            BuilderType::env_builder(env_builder, n_envs),
            rollout_mode,
        )
    }
}
