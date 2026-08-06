use std::io::Write;
use std::{fs::File, marker::PhantomData, sync::mpsc::Sender, time::Instant};

use burn::{
    grad_clipping::GradientClippingConfig, optim::AdamWConfig, prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::{
    a2c::{A2C, A2CHook, A2CParams},
    ppo::{PPO, PPOHook, PPOParams},
};
use r2l_burn::learning_module::ActionSpacePolicyValueModule as BurnPolicyValueModule;
use r2l_candle::learning_module::PolicyValueModule as CandlePolicyValueModule;
use r2l_core::env::EnvDescription;
use r2l_core::env::normalizer::{ClippedNormalizer, ClippedNormalizerInner, NormalizerMode};
use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    models::ActivationFunction,
    on_policy::{
        algorithm::{Agent, OnPolicyAlgorithm, OnPolicyRuntime, Sampler},
        learning_module::OnPolicyLearningModule,
    },
    rng::set_seed,
    tensor::R2lTensor,
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{
    DirectSampler, DirectSamplerCore, SamplerExecutionMode, StagedSampler, StagedSamplerCore,
};

use crate::evaluators::best_actor_evaluator::EvaluationSampler;
use crate::utils::RewardNormalizer;
use crate::{
    BestActorEvaluator, BurnBackend, BurnBackendConfig, CandleBackend,
    DefaultOnPolicyAlgorithmHooks, InferenceBackend, InferenceConfig, InferenceObservationMode,
    LearningRateSchedule, LearningSchedule, OnPolicyCommandReceiver, PolicyBuilder,
    TrainingArtifactsConfig,
    builders::learning_module::{AdamWParams, OnPolicyOptimizerLayout},
    hooks::{
        a2c::{A2CStats, DefaultA2CHook, DefaultA2CHookReporter},
        on_policy::PerformanceLog,
        ppo::{DefaultPPOHook, DefaultPPOHookReporter, PPOStats, TargetKl},
    },
};
use crate::{EpisodeBoundHook, StepBoundHook};

const PERFORMANCE_FILE: &str = "performance.csv";

pub type PPOCandle = PPO<CandlePolicyValueModule, DefaultPPOHook<CandlePolicyValueModule>>;
pub type PPOBurn<B> = PPO<BurnPolicyValueModule<B>, DefaultPPOHook<BurnPolicyValueModule<B>>>;
pub type A2CCandle = A2C<CandlePolicyValueModule, DefaultA2CHook<CandlePolicyValueModule>>;
pub type A2CBurn<B> = A2C<BurnPolicyValueModule<B>, DefaultA2CHook<BurnPolicyValueModule<B>>>;

enum SamplerConfiguration<E: Env> {
    Direct,
    Staged {
        clipped_normalizer_inner: Option<ClippedNormalizerInner<E::Tensor>>,
    },
}

trait EnvBuildPlan<E: Env> {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: usize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> EvaluationSampler<E>;

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<E>;

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> StagedSamplerCore<E>;
}

struct TypedEnvBuildPlan<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
}

impl<EB: EnvBuilder<Env: Env>> EnvBuildPlan<EB::Env> for TypedEnvBuildPlan<EB> {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: usize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
    ) -> EvaluationSampler<EB::Env> {
        EvaluationSampler::build(
            self.env_builder.clone(),
            episodes_per_evaluation,
            evaluation_execution_mode,
            obs_normalizer,
        )
    }

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<EB::Env> {
        DirectSamplerCore::build(self.env_builder.clone(), execution_mode)
    }

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
    ) -> StagedSamplerCore<EB::Env> {
        StagedSamplerCore::build(self.env_builder.clone(), execution_mode, obs_normalizer)
    }
}

struct LearningModuleConfiguration {
    policy_builder: PolicyBuilder,
    value_hidden_layers: Vec<usize>,
    optimizer_layout: OnPolicyOptimizerLayout,
}

impl Default for LearningModuleConfiguration {
    fn default() -> Self {
        Self {
            policy_builder: PolicyBuilder::default(),
            value_hidden_layers: vec![64, 64],
            optimizer_layout: OnPolicyOptimizerLayout::Joint {
                params: AdamWParams {
                    lr: 3e-4,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-5,
                    weight_decay: 1e-4,
                },
                max_grad_norm: None,
            },
        }
    }
}

impl LearningModuleConfiguration {
    fn build_candle<T: R2lTensor>(
        self,
        observation_size: usize,
        action_space: r2l_core::env::Space<T>,
        device: &Device,
    ) -> anyhow::Result<CandlePolicyValueModule> {
        let (policy, policy_varmap) =
            self.policy_builder
                .build_candle_with_varmap(observation_size, action_space, device)?;
        let activation_function = self.policy_builder.activation_function;
        match self.optimizer_layout {
            OnPolicyOptimizerLayout::Joint {
                max_grad_norm,
                params,
            } => CandlePolicyValueModule::build_joint(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                max_grad_norm,
                Self::candle_optimizer_params(params),
                activation_function,
            ),
            OnPolicyOptimizerLayout::Split {
                policy_max_grad_norm,
                policy_params,
                value_max_grad_norm,
                value_params,
            } => CandlePolicyValueModule::build_split(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                policy_max_grad_norm,
                value_max_grad_norm,
                Self::candle_optimizer_params(policy_params),
                Self::candle_optimizer_params(value_params),
                activation_function,
            ),
        }
    }

    fn build_burn<B: AutodiffBackend, T: R2lTensor>(
        self,
        observation_size: usize,
        action_space: r2l_core::env::Space<T>,
    ) -> BurnPolicyValueModule<B> {
        let policy = self
            .policy_builder
            .build_burn::<B, _>(observation_size, action_space);
        let activation_function = self.policy_builder.activation_function;
        let value_layers = &[&[observation_size][..], &self.value_hidden_layers, &[1]].concat();
        match self.optimizer_layout {
            OnPolicyOptimizerLayout::Joint {
                max_grad_norm,
                params,
            } => {
                let optimizer_config = Self::burn_optimizer_config(&params, max_grad_norm);
                BurnPolicyValueModule::joint(
                    policy,
                    value_layers,
                    activation_function,
                    optimizer_config,
                    params.lr,
                )
            }
            OnPolicyOptimizerLayout::Split {
                policy_max_grad_norm,
                policy_params,
                value_max_grad_norm,
                value_params,
            } => BurnPolicyValueModule::split(
                policy,
                value_layers,
                activation_function,
                Self::burn_optimizer_config(&policy_params, policy_max_grad_norm),
                policy_params.lr,
                Self::burn_optimizer_config(&value_params, value_max_grad_norm),
                value_params.lr,
            ),
        }
    }

    fn candle_optimizer_params(params: AdamWParams) -> ParamsAdamW {
        ParamsAdamW {
            lr: params.lr,
            beta1: params.beta1,
            beta2: params.beta2,
            eps: params.eps,
            weight_decay: params.weight_decay,
        }
    }

    fn burn_optimizer_config(params: &AdamWParams, max_grad_norm: Option<f32>) -> AdamWConfig {
        let optimizer_config = AdamWConfig::new()
            .with_beta_1(params.beta1 as f32)
            .with_beta_2(params.beta2 as f32)
            .with_epsilon(params.eps as f32)
            .with_weight_decay(params.weight_decay as f32);
        match max_grad_norm {
            Some(max_grad_norm) => optimizer_config
                .with_grad_clipping(Some(GradientClippingConfig::Norm(max_grad_norm))),
            None => optimizer_config,
        }
    }
}

struct Builder<E: Env> {
    env_build_plan: Box<dyn EnvBuildPlan<E>>,
    env_desription: EnvDescription<E::Tensor>,
    n_envs: usize,
    sampler_configuraion: SamplerConfiguration<E>,

    // for the hooks
    learning_schedule: LearningSchedule,
    learning_rate_schedule: Option<LearningRateSchedule>,
    training_artifacts_config: Option<TrainingArtifactsConfig>,
    policy_command_rx: Option<OnPolicyCommandReceiver>,

    // for the agent
    learning_module_configuration: Option<LearningModuleConfiguration>,
    normalize_advantage: Option<bool>,
    log_progress: bool,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    gradient_clipping: Option<f32>,
    gamma: f32,
    lambda: f32,
    sample_size: usize,
    total_epochs: usize,
    target_kl: Option<f32>,
    clip_range: f32,
    ppo_reporter: Option<Sender<PPOStats>>,
    a2c_reporter: Option<Sender<A2CStats>>,
    candle_backend: Option<CandleBackend>,
    burn_backend: Option<BurnBackendConfig>,
    seed: Option<u64>,

    // for the sampler
    sampler_execution_mode: SamplerExecutionMode,
    reward_normalizer: Option<RewardNormalizer>,
    rollout_steps: usize,
    rollout_episodes: usize,
}

impl<E: Env> Builder<E> {
    fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, n_envs: usize) -> Self {
        let env_desription = env_builder.env_description().unwrap();
        Self {
            env_build_plan: Box::new(TypedEnvBuildPlan {
                env_builder: EnvBuilderType::homogeneous(env_builder, n_envs),
            }),
            env_desription,
            n_envs,
            sampler_configuraion: SamplerConfiguration::Direct,
            learning_schedule: LearningSchedule::rollout_bound(300),
            learning_rate_schedule: None,
            training_artifacts_config: None,
            policy_command_rx: None,
            learning_module_configuration: Some(LearningModuleConfiguration::default()),
            normalize_advantage: None,
            log_progress: true,
            entropy_coeff: 0.0,
            vf_coeff: None,
            gradient_clipping: None,
            gamma: 0.98,
            lambda: 0.8,
            sample_size: 64,
            total_epochs: 10,
            target_kl: None,
            clip_range: 0.2,
            ppo_reporter: None,
            a2c_reporter: None,
            candle_backend: Some(CandleBackend {
                device: Device::Cpu,
            }),
            burn_backend: Some(BurnBackendConfig),
            seed: None,
            sampler_execution_mode: SamplerExecutionMode::MultiThreaded,
            reward_normalizer: None,
            rollout_steps: 1024,
            rollout_episodes: 1,
        }
    }

    fn obs_normalizer(
        &self,
        normalizer_mode: NormalizerMode,
    ) -> Option<ClippedNormalizer<E::Tensor>> {
        let SamplerConfiguration::Staged {
            clipped_normalizer_inner: Some(inner),
        } = &self.sampler_configuraion
        else {
            return None;
        };
        // TODO: this can error, but it's fine for now! In fact, catching this through the test
        // suite would be nice!
        let normalizer = ClippedNormalizer {
            normalizer_mode,
            inner: inner.clone(),
        };
        Some(normalizer)
    }

    fn update_optimizer_layout(
        &mut self,
        update: impl FnOnce(OnPolicyOptimizerLayout) -> OnPolicyOptimizerLayout,
    ) {
        let LearningModuleConfiguration {
            policy_builder,
            value_hidden_layers,
            optimizer_layout,
        } = self.learning_module_configuration.take().unwrap();
        self.learning_module_configuration = Some(LearningModuleConfiguration {
            policy_builder,
            value_hidden_layers,
            optimizer_layout: update(optimizer_layout),
        });
    }

    fn evaluator<A: Agent>(
        &mut self,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Option<BestActorEvaluator<A::Actor, E>> {
        let config = self.training_artifacts_config.take()?;
        if !config.evaluation_results && !config.inference_artifacts {
            return None;
        }
        let evaluation_sampler = self.env_build_plan.build_evaluator_sampler(
            config.evaluation_settings.episodes_per_evaluation,
            config.evaluation_settings.evaluation_execution_mode,
            obs_normalizer,
        );
        Some(config.build_with_sampler(evaluation_sampler))
    }

    fn default_on_policy_hook<A: Agent, S: Sampler<Tensor = E::Tensor>>(
        mut self,
    ) -> DefaultOnPolicyAlgorithmHooks<A, S, E> {
        let performance_log = self.training_artifacts_config.as_ref().map(|config| -> _ {
            let output_dir = config.output_dir.clone();
            std::fs::create_dir_all(&output_dir).unwrap();
            let mut file = File::create(output_dir.join(PERFORMANCE_FILE)).unwrap();
            writeln!(
                file,
                "rollout,collect_ms,learn_ms,evaluate_ms,rollout_ms,total_ms"
            )
            .unwrap();
            let now = Instant::now();
            PerformanceLog {
                file,
                training_started: now,
                rollout_started: now,
                phase_started: now,
                collect_ms: 0.0,
                rollout: 0,
            }
        });
        let obs_normalizer = self.obs_normalizer(NormalizerMode::ReadOnly);
        let evaluator = self.evaluator::<A>(obs_normalizer);
        DefaultOnPolicyAlgorithmHooks {
            learning_schedule: self.learning_schedule,
            learning_rate_schedule: self.learning_rate_schedule,
            evaluator,
            performance_log,
            command_rx: self.policy_command_rx.take(),
            _phantom: PhantomData,
        }
    }

    fn direct_sampler_step_bound(&self) -> DirectSampler<E, StepBoundHook<E>> {
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.sampler_execution_mode);
        let reward_normalizer = self.reward_normalizer.clone();
        let step_bound_hook = StepBoundHook::new(self.rollout_steps, reward_normalizer);
        DirectSampler::new(sampler_core, step_bound_hook)
    }

    fn direct_sampler_episode_bound(&self) -> DirectSampler<E, EpisodeBoundHook<E>> {
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.sampler_execution_mode);
        let episode_bound_hook = EpisodeBoundHook::new(self.rollout_episodes);
        DirectSampler::new(sampler_core, episode_bound_hook)
    }

    fn staged_sampler_step_bound(&self) -> StagedSampler<E, StepBoundHook<E>> {
        let obs_normalizer = self.obs_normalizer(NormalizerMode::Update);
        let sampler_core = self
            .env_build_plan
            .build_staged_sampler_core(self.sampler_execution_mode, obs_normalizer);
        let reward_normalizer = self.reward_normalizer.clone();
        let step_bound_hook = StepBoundHook::new(self.rollout_steps, reward_normalizer);
        StagedSampler {
            core: sampler_core,
            hook: step_bound_hook,
        }
    }

    fn write_inference_config(&self, backend: InferenceBackend) -> anyhow::Result<()> {
        if let Some(config) = &self.training_artifacts_config
            && config.inference_artifacts
        {
            let observation_mode = match &self.sampler_configuraion {
                SamplerConfiguration::Staged {
                    clipped_normalizer_inner: Some(_),
                } => InferenceObservationMode::Normalized,
                _ => InferenceObservationMode::Raw,
            };
            let policy_builder = self
                .learning_module_configuration
                .as_ref()
                .unwrap()
                .policy_builder
                .clone();
            InferenceConfig::new(policy_builder, observation_mode, backend)
                .write_to_dir(&config.output_dir)?;
        }
        Ok(())
    }

    fn ppo_hook<M>(&mut self) -> DefaultPPOHook<M> {
        DefaultPPOHook {
            normalize_advantage: self.normalize_advantage.unwrap_or(true),
            total_epochs: self.total_epochs,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            target_kl: self.target_kl.map(|target| TargetKl {
                target,
                target_exceeded: false,
            }),
            gradient_clipping: self.gradient_clipping,
            current_epoch: 0,
            reporter: DefaultPPOHookReporter::new(
                self.ppo_reporter.take(),
                self.log_progress,
                self.n_envs,
            ),
            rollout_idx: 0,
            _lm: PhantomData,
        }
    }

    fn a2c_hook<M>(&mut self) -> DefaultA2CHook<M> {
        DefaultA2CHook {
            normalize_advantage: self.normalize_advantage.unwrap_or(false),
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            gradient_clipping: self.gradient_clipping,
            reporter: DefaultA2CHookReporter::new(
                self.a2c_reporter.take(),
                self.log_progress,
                self.n_envs,
            ),
            _lm: PhantomData,
        }
    }

    fn ppo_params(&self) -> PPOParams {
        PPOParams {
            clip_range: self.clip_range,
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size,
        }
    }

    fn a2c_params(&self) -> A2CParams {
        A2CParams {
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size,
        }
    }

    fn ppo_candle_agent(&mut self) -> anyhow::Result<PPOCandle> {
        let backend = self.candle_backend.take().unwrap();
        self.write_inference_config(InferenceBackend::Candle(backend.clone()))?;
        if let Some(seed) = self.seed {
            backend.seed(seed);
        }
        let learning_module = self
            .learning_module_configuration
            .take()
            .unwrap()
            .build_candle(
                self.env_desription.observation_size(),
                self.env_desription.action_space.clone(),
                &backend.device,
            )?;
        let hooks = self.ppo_hook();
        Ok(PPO {
            lm: learning_module,
            hooks,
            params: self.ppo_params(),
        })
    }

    fn ppo_burn_agent(&mut self) -> anyhow::Result<PPOBurn<BurnBackend>> {
        let backend = self.burn_backend.take().unwrap();
        self.write_inference_config(InferenceBackend::Burn(backend))?;
        if let Some(seed) = self.seed {
            BurnBackend::seed(&Default::default(), seed);
        }
        let learning_module = self
            .learning_module_configuration
            .take()
            .unwrap()
            .build_burn::<BurnBackend, _>(
                self.env_desription.observation_size(),
                self.env_desription.action_space.clone(),
            );
        let hooks = self.ppo_hook();
        Ok(PPO {
            lm: learning_module,
            hooks,
            params: self.ppo_params(),
        })
    }

    fn a2c_candle_agent(&mut self) -> anyhow::Result<A2CCandle> {
        let backend = self.candle_backend.take().unwrap();
        self.write_inference_config(InferenceBackend::Candle(backend.clone()))?;
        if let Some(seed) = self.seed {
            backend.seed(seed);
        }
        let learning_module = self
            .learning_module_configuration
            .take()
            .unwrap()
            .build_candle(
                self.env_desription.observation_size(),
                self.env_desription.action_space.clone(),
                &backend.device,
            )?;
        let hooks = self.a2c_hook();
        Ok(A2C {
            lm: learning_module,
            hooks,
            params: self.a2c_params(),
        })
    }

    fn a2c_burn_agent(&mut self) -> anyhow::Result<A2CBurn<BurnBackend>> {
        let backend = self.burn_backend.take().unwrap();
        self.write_inference_config(InferenceBackend::Burn(backend))?;
        if let Some(seed) = self.seed {
            BurnBackend::seed(&Default::default(), seed);
        }
        let learning_module = self
            .learning_module_configuration
            .take()
            .unwrap()
            .build_burn::<BurnBackend, _>(
                self.env_desription.observation_size(),
                self.env_desription.action_space.clone(),
            );
        let hooks = self.a2c_hook();
        Ok(A2C {
            lm: learning_module,
            hooks,
            params: self.a2c_params(),
        })
    }
}

trait Buildable<E: Env> {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self>
    where
        Self: Sized;
}

struct Config<A: Agent, S: Sampler, E: Env>(PhantomData<(A, S, E)>);

pub struct OnPolicyAlgoBuilder<A: Agent, S: Sampler, E: Env> {
    builder: Builder<E>,
    _config: Config<A, S, E>,
}

impl<A: Agent, S: Sampler, E: Env> OnPolicyAlgoBuilder<A, S, E> {
    pub fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, n_envs: usize) -> Self {
        Self {
            builder: Builder::new(env_builder, n_envs),
            _config: Config(PhantomData),
        }
    }

    fn with_agent<A2: Agent>(self) -> OnPolicyAlgoBuilder<A2, S, E> {
        OnPolicyAlgoBuilder {
            builder: self.builder,
            _config: Config(PhantomData),
        }
    }

    pub fn with_training_artifacts(mut self, config: TrainingArtifactsConfig) -> Self {
        self.builder.training_artifacts_config = Some(config);
        self
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.builder.learning_schedule = learning_schedule;
        self
    }

    pub fn with_command_rx(mut self, command_rx: OnPolicyCommandReceiver) -> Self {
        self.builder.policy_command_rx = Some(command_rx);
        self
    }

    pub fn with_learning_rate_schedule(
        mut self,
        learning_rate_schedule: Option<LearningRateSchedule>,
    ) -> Self {
        self.builder.learning_rate_schedule = learning_rate_schedule;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.builder.seed = Some(seed);
        self
    }

    pub fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.builder.sampler_execution_mode = execution_mode;
        self
    }

    pub fn with_policy_builder(mut self, policy_builder: PolicyBuilder) -> Self {
        self.builder
            .learning_module_configuration
            .as_mut()
            .unwrap()
            .policy_builder = policy_builder;
        self
    }

    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.builder
            .learning_module_configuration
            .as_mut()
            .unwrap()
            .policy_builder
            .hidden_layers = policy_hidden_layers;
        self
    }

    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.builder
            .learning_module_configuration
            .as_mut()
            .unwrap()
            .policy_builder
            .activation_function = activation_function;
        self
    }

    pub fn with_log_std_init(mut self, log_std_init: f32) -> Self {
        self.builder
            .learning_module_configuration
            .as_mut()
            .unwrap()
            .policy_builder
            .log_std_init = log_std_init;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_lr(learning_rate));
        self.builder.learning_rate_schedule = Some(LearningRateSchedule::Constant(learning_rate));
        self
    }

    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_beta1(beta1));
        self
    }

    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_beta2(beta2));
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_epsilon(epsilon));
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_weight_decay(weight_decay));
        self
    }

    pub fn with_joint(mut self, max_grad_norm: Option<f32>, params: AdamWParams) -> Self {
        self.builder
            .update_optimizer_layout(|_| OnPolicyOptimizerLayout::Joint {
                max_grad_norm,
                params,
            });
        self
    }

    pub fn with_split(
        mut self,
        policy_max_grad_norm: Option<f32>,
        policy_params: AdamWParams,
        value_max_grad_norm: Option<f32>,
        value_params: AdamWParams,
    ) -> Self {
        self.builder
            .update_optimizer_layout(|_| OnPolicyOptimizerLayout::Split {
                policy_max_grad_norm,
                policy_params,
                value_max_grad_norm,
                value_params,
            });
        self
    }

    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.builder
            .learning_module_configuration
            .as_mut()
            .unwrap()
            .value_hidden_layers = value_hidden_layers;
        self
    }

    pub fn with_optimizer_layout(mut self, optimizer_layout: OnPolicyOptimizerLayout) -> Self {
        self.builder.update_optimizer_layout(|_| optimizer_layout);
        self
    }

    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.builder.normalize_advantage = Some(normalize_advantage);
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.builder.entropy_coeff = entropy_coeff;
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.builder.vf_coeff = vf_coeff;
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.builder.gradient_clipping = gradient_clipping;
        self
    }

    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.builder.log_progress = log_progress;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.builder.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.builder.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.builder.sample_size = sample_size;
        self
    }
}

impl<S: Sampler, E: Env> OnPolicyAlgoBuilder<PPOCandle, S, E> {
    pub fn with_candle(mut self, device: Device) -> Self {
        self.builder.candle_backend = Some(CandleBackend { device });
        self
    }

    pub fn with_burn(mut self) -> OnPolicyAlgoBuilder<PPOBurn<BurnBackend>, S, E> {
        self.builder.burn_backend = Some(BurnBackendConfig);
        self.with_agent()
    }
}

impl<S: Sampler, E: Env> OnPolicyAlgoBuilder<PPOBurn<BurnBackend>, S, E> {
    pub fn with_candle(mut self, device: Device) -> OnPolicyAlgoBuilder<PPOCandle, S, E> {
        self.builder.candle_backend = Some(CandleBackend { device });
        self.with_agent()
    }

    pub fn with_burn(mut self) -> Self {
        self.builder.burn_backend = Some(BurnBackendConfig);
        self
    }
}

impl<M, S, E> OnPolicyAlgoBuilder<PPO<M, DefaultPPOHook<M>>, S, E>
where
    M: OnPolicyLearningModule,
    DefaultPPOHook<M>: PPOHook<M>,
    S: Sampler,
    E: Env,
{
    pub fn with_reporter(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.builder.ppo_reporter = tx;
        self
    }

    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.builder.total_epochs = total_epochs;
        self
    }

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.builder.target_kl = target_kl;
        self
    }

    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.builder.clip_range = clip_range;
        self
    }
}

impl<S: Sampler, E: Env> OnPolicyAlgoBuilder<A2CCandle, S, E> {
    pub fn with_candle(mut self, device: Device) -> Self {
        self.builder.candle_backend = Some(CandleBackend { device });
        self
    }

    pub fn with_burn(mut self) -> OnPolicyAlgoBuilder<A2CBurn<BurnBackend>, S, E> {
        self.builder.burn_backend = Some(BurnBackendConfig);
        self.with_agent()
    }
}

impl<S: Sampler, E: Env> OnPolicyAlgoBuilder<A2CBurn<BurnBackend>, S, E> {
    pub fn with_candle(mut self, device: Device) -> OnPolicyAlgoBuilder<A2CCandle, S, E> {
        self.builder.candle_backend = Some(CandleBackend { device });
        self.with_agent()
    }

    pub fn with_burn(mut self) -> Self {
        self.builder.burn_backend = Some(BurnBackendConfig);
        self
    }
}

impl<M, S, E> OnPolicyAlgoBuilder<A2C<M, DefaultA2CHook<M>>, S, E>
where
    M: OnPolicyLearningModule,
    DefaultA2CHook<M>: A2CHook<M>,
    S: Sampler,
    E: Env,
{
    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.builder.a2c_reporter = tx;
        self
    }
}

impl<A: Agent, E: Env> OnPolicyAlgoBuilder<A, DirectSampler<E, StepBoundHook<E>>, E> {
    pub fn with_rollout_steps(mut self, rollout_steps: usize) -> Self {
        self.builder.rollout_steps = rollout_steps;
        self
    }

    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        self.builder.reward_normalizer = Some(RewardNormalizer::new(
            self.builder.n_envs,
            gamma,
            clip_reward,
        ));
        self
    }
}

impl<A: Agent, E: Env> OnPolicyAlgoBuilder<A, DirectSampler<E, EpisodeBoundHook<E>>, E> {
    pub fn with_rollout_episodes(mut self, rollout_episodes: usize) -> Self {
        self.builder.rollout_episodes = rollout_episodes;
        self
    }
}

impl<A: Agent, E: Env> OnPolicyAlgoBuilder<A, StagedSampler<E, StepBoundHook<E>>, E> {
    pub fn with_rollout_steps(mut self, rollout_steps: usize) -> Self {
        self.builder.rollout_steps = rollout_steps;
        self
    }

    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        self.builder.reward_normalizer = Some(RewardNormalizer::new(
            self.builder.n_envs,
            gamma,
            clip_reward,
        ));
        self
    }

    pub fn with_observation_normalizer(mut self, obs_clip: Option<f32>) -> Self {
        let clipped_normalizer_inner = obs_clip.map(|clip| {
            ClippedNormalizer::build(
                NormalizerMode::Update,
                clip,
                vec![self.builder.env_desription.observation_space.size()],
            )
            .inner
        });
        self.builder.sampler_configuraion = SamplerConfiguration::Staged {
            clipped_normalizer_inner,
        };
        self
    }
}

impl<E: Env> Buildable<E> for PPOCandle {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self> {
        builder.ppo_candle_agent()
    }
}

impl<E: Env> Buildable<E> for A2CCandle {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self> {
        builder.a2c_candle_agent()
    }
}

impl<E: Env> Buildable<E> for PPOBurn<BurnBackend> {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self> {
        builder.ppo_burn_agent()
    }
}

impl<E: Env> Buildable<E> for A2CBurn<BurnBackend> {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self> {
        builder.a2c_burn_agent()
    }
}

impl<E: Env> Buildable<E> for DirectSampler<E, StepBoundHook<E>> {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self> {
        Ok(builder.direct_sampler_step_bound())
    }
}

impl<E: Env> Buildable<E> for DirectSampler<E, EpisodeBoundHook<E>> {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self> {
        Ok(builder.direct_sampler_episode_bound())
    }
}

impl<E: Env> Buildable<E> for StagedSampler<E, StepBoundHook<E>> {
    fn build(builder: &mut Builder<E>) -> anyhow::Result<Self> {
        Ok(builder.staged_sampler_step_bound())
    }
}

impl<A: Agent + Buildable<E>, S: Sampler + Buildable<E>, E: Env<Tensor = S::Tensor>>
    OnPolicyAlgoBuilder<A, S, E>
{
    pub fn build(
        mut self,
    ) -> anyhow::Result<OnPolicyAlgorithm<A, S, DefaultOnPolicyAlgorithmHooks<A, S, E>>> {
        if let Some(seed) = self.builder.seed {
            set_seed(seed);
        }
        let agent = A::build(&mut self.builder)?;
        let sampler = S::build(&mut self.builder)?;
        let hooks = self.builder.default_on_policy_hook();
        Ok(OnPolicyAlgorithm::new(
            OnPolicyRuntime { agent, sampler },
            hooks,
        ))
    }
}

pub type PPO2AlgorithmBuilder<E> =
    OnPolicyAlgoBuilder<PPOCandle, DirectSampler<E, StepBoundHook<E>>, E>;

impl PPO2AlgorithmBuilder<GymEnv> {
    pub fn gym<EB: Into<GymEnvBuilder>>(env_builder: EB, n_envs: usize) -> Self {
        Self::new(env_builder.into(), n_envs)
    }
}
