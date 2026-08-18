pub(crate) mod inference;
pub(crate) mod normalizer;
pub(crate) mod policy;

use std::{marker::PhantomData, path::PathBuf, sync::mpsc::Sender};

use burn::{
    backend::ndarray::NdArrayDevice, grad_clipping::GradientClippingConfig, optim::AdamWConfig,
    prelude::Backend, tensor::backend::AutodiffBackend,
};
use candle_core::{Device, DeviceLocation};
use candle_nn::ParamsAdamW;
pub use inference::InferenceRunner;
use inference::{InferenceBackend, InferenceConfig, InferenceObservationMode};
use policy::PolicyBuilder;
use r2l_agents::on_policy_algorithms::{
    a2c::{A2C, A2CHook, A2CParams},
    ppo::{PPO, PPOHook, PPOParams},
};
use r2l_burn::learning_module::PolicyValueLearner as BurnPolicyValueLearner;
use r2l_candle::learning_module::PolicyValueLearner as CandlePolicyValueLearner;
use r2l_core::env::EnvDescription;
use r2l_core::env::normalizer::{ClippedNormalizer, NormalizerMode};
use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    error::Error,
    models::{ActivationFunction, ToSafetensors},
    on_policy::{
        algorithm::{Agent, OnPolicyAlgorithm, OnPolicyRuntime, Sampler},
        learning_module::OnPolicyLearner,
    },
    rng::set_seed,
};
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{
    DirectSampler, DirectSamplerCore, SamplerExecutionMode, StagedSampler, StagedSamplerCore,
};
use serde::{Deserialize, Serialize, de::Error as _};

use crate::hooks::on_policy::OnPolicyTrainingHooks;
use crate::{
    A2CRolloutStats, PPORolloutStats,
    evaluator::{EvaluationSampler, EvaluationSettings},
    hooks::{
        a2c::A2CRolloutReporter,
        on_policy::{
            OnPolicyCommandHandler, OnPolicyControlEndpoint, ScheduledEvaluator,
            TrainingTimingRecorder, on_policy_control_channel,
        },
        ppo::TargetKl,
    },
};
use crate::{BurnBackend, LearningRateSchedule, OnPolicyControlHandle, TrainingLimit};
use crate::{EpisodeBoundHook, StepBoundHook};
use crate::{
    evaluator::BestActorEvaluator,
    hooks::{a2c::A2CLearningHook, ppo::PPOLearningHook},
};
use crate::{hooks::ppo::PPORolloutReporter, utils::RewardNormalizer};

/// PPO agent produced by a Candle-backed algorithm builder.
pub type PPOCandle = PPO<CandlePolicyValueLearner, PPOLearningHook<CandlePolicyValueLearner>>;
/// PPO agent produced by a Burn-backed algorithm builder.
pub type PPOBurn<B> = PPO<BurnPolicyValueLearner<B>, PPOLearningHook<BurnPolicyValueLearner<B>>>;
/// A2C agent produced by a Candle-backed algorithm builder.
pub type A2CCandle = A2C<CandlePolicyValueLearner, A2CLearningHook<CandlePolicyValueLearner>>;
/// A2C agent produced by a Burn-backed algorithm builder.
pub type A2CBurn<B> = A2C<BurnPolicyValueLearner<B>, A2CLearningHook<BurnPolicyValueLearner<B>>>;

/// Hyperparameters for Adam optimization with decoupled weight decay.
#[derive(Clone, Debug)]
pub struct AdamWParams {
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay coefficient.
    pub beta1: f64,
    /// Second-moment decay coefficient.
    pub beta2: f64,
    /// Numerical-stability term.
    pub eps: f64,
    /// Weight-decay coefficient.
    pub weight_decay: f64,
}

/// Selects the artifacts produced during training and where they are written.
pub struct TrainingArtifactsConfig {
    pub(crate) output_dir: PathBuf,
    pub(crate) evaluation_results: bool,
    pub(crate) training_timings: bool,
    pub(crate) inference_artifacts: bool,
    pub(crate) evaluation_settings: EvaluationSettings,
}

impl TrainingArtifactsConfig {
    /// Creates a configuration that writes all supported training artifacts.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory in which training artifacts will be written.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: resolve_and_validate_output_dir(output_dir.into()),
            evaluation_results: true,
            training_timings: true,
            inference_artifacts: true,
            evaluation_settings: EvaluationSettings::default(),
        }
    }

    /// Sets whether evaluation results are written during training.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to write evaluation results.
    #[must_use]
    pub fn with_evaluation_results(mut self, enabled: bool) -> Self {
        self.evaluation_results = enabled;
        self
    }

    /// Sets whether training timing measurements are written.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to write training timing measurements.
    #[must_use]
    pub fn with_training_timings(mut self, enabled: bool) -> Self {
        self.training_timings = enabled;
        self
    }

    /// Sets whether the best policy is saved as inference-ready artifacts.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to save inference-ready artifacts for the best policy.
    #[must_use]
    pub fn with_inference_artifacts(mut self, enabled: bool) -> Self {
        self.inference_artifacts = enabled;
        self
    }

    /// Sets the evaluation behavior used by evaluation results and inference artifacts.
    ///
    /// # Arguments
    ///
    /// * `evaluation_settings` - Settings that control evaluation frequency and execution.
    #[must_use]
    pub fn with_evaluation_settings(mut self, evaluation_settings: EvaluationSettings) -> Self {
        self.evaluation_settings = evaluation_settings;
        self
    }

    fn needs_evaluator(&self) -> bool {
        self.evaluation_results || self.inference_artifacts
    }

    fn needs_timing_recorder(&self) -> bool {
        self.training_timings
    }
}

fn resolve_and_validate_output_dir(path: PathBuf) -> PathBuf {
    let path = if path.is_absolute() {
        path
    } else {
        std::env::current_dir().unwrap().join(path)
    };
    assert!(!path.is_file());
    path
}

/// Optimizer arrangement for the policy and value networks.
#[derive(Clone, Debug)]
pub(crate) enum OnPolicyOptimizerLayout {
    /// One optimizer updates both networks.
    Joint {
        /// Optional maximum gradient norm.
        max_grad_norm: Option<f32>,
        /// Shared `AdamW` parameters.
        params: AdamWParams,
    },
    /// Policy and value networks use independent optimizers.
    Split {
        /// Optional maximum policy gradient norm.
        policy_max_grad_norm: Option<f32>,
        /// Policy optimizer parameters.
        policy_params: AdamWParams,
        /// Optional maximum value gradient norm.
        value_max_grad_norm: Option<f32>,
        /// Value optimizer parameters.
        value_params: AdamWParams,
    },
}

impl OnPolicyOptimizerLayout {
    fn map_params(mut self, mut update: impl FnMut(&mut AdamWParams)) -> Self {
        match &mut self {
            Self::Joint { params, .. } => update(params),
            Self::Split {
                policy_params,
                value_params,
                ..
            } => {
                update(policy_params);
                update(value_params);
            }
        }
        self
    }

    /// Sets the learning rate of every optimizer in the layout.
    #[must_use]
    fn with_lr(self, lr: f64) -> Self {
        self.map_params(|params| params.lr = lr)
    }

    /// Sets the first-moment decay of every optimizer in the layout.
    #[must_use]
    fn with_beta1(self, beta1: f64) -> Self {
        self.map_params(|params| params.beta1 = beta1)
    }

    /// Sets the second-moment decay of every optimizer in the layout.
    #[must_use]
    fn with_beta2(self, beta2: f64) -> Self {
        self.map_params(|params| params.beta2 = beta2)
    }

    /// Sets the numerical-stability term of every optimizer in the layout.
    #[must_use]
    fn with_epsilon(self, epsilon: f64) -> Self {
        self.map_params(|params| params.eps = epsilon)
    }

    /// Sets the weight decay of every optimizer in the layout.
    #[must_use]
    fn with_weight_decay(self, weight_decay: f64) -> Self {
        self.map_params(|params| params.weight_decay = weight_decay)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub(crate) struct BurnBackendConfig;

#[derive(Debug, Clone)]
pub(crate) struct CandleBackend {
    pub(crate) device: Device,
}

#[derive(Serialize, Deserialize)]
enum CandleDeviceConfig {
    Cpu,
    Cuda { ordinal: usize },
    Metal { ordinal: usize },
}

impl Serialize for CandleBackend {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let device = match self.device.location() {
            DeviceLocation::Cpu => CandleDeviceConfig::Cpu,
            DeviceLocation::Cuda { gpu_id } => CandleDeviceConfig::Cuda { ordinal: gpu_id },
            DeviceLocation::Metal { gpu_id } => CandleDeviceConfig::Metal { ordinal: gpu_id },
        };
        device.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CandleBackend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let device = match CandleDeviceConfig::deserialize(deserializer)? {
            CandleDeviceConfig::Cpu => Device::Cpu,
            CandleDeviceConfig::Cuda { ordinal } => {
                Device::new_cuda(ordinal).map_err(D::Error::custom)?
            }
            CandleDeviceConfig::Metal { ordinal } => {
                Device::new_metal(ordinal).map_err(D::Error::custom)?
            }
        };
        Ok(Self { device })
    }
}

impl CandleBackend {
    fn seed(&self, seed: u64) -> Result<(), Error> {
        if !matches!(&self.device, Device::Cpu) {
            self.device.set_seed(seed).map_err(Error::wrap)?;
        }
        Ok(())
    }
}

enum SamplerConfiguration<E: Env> {
    DirectStep {
        rollout_steps: usize,
        reward_normalizer: Option<RewardNormalizer>,
    },
    DirectEpisode {
        rollout_episodes: usize,
    },
    StagedStep {
        rollout_steps: usize,
        reward_normalizer: Option<RewardNormalizer>,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    },
}

impl<E: Env> SamplerConfiguration<E> {
    fn obs_normalizer(&self) -> Option<ClippedNormalizer<E::Tensor>> {
        match self {
            Self::StagedStep {
                obs_normalizer: Some(obs_normalizer),
                ..
            } => Some(obs_normalizer.clone()),
            _ => None,
        }
    }
}

enum BackendConfiguration {
    Candle(CandleBackend),
    Burn(BurnBackendConfig),
}

enum AlgorithmConfiguration {
    Ppo {
        normalize_advantage: Option<bool>,
        total_epochs: usize,
        target_kl: Option<f32>,
        clip_range: f32,
        reporter: Option<Sender<PPORolloutStats>>,
    },
    A2C {
        normalize_advantage: Option<bool>,
        reporter: Option<Sender<A2CRolloutStats>>,
    },
}

trait EnvBuildPlan<E: Env>: Send {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: usize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Result<EvaluationSampler<E>, Error>;

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<E>;

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Result<StagedSamplerCore<E>, Error>;
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
    ) -> Result<EvaluationSampler<EB::Env>, Error> {
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
    ) -> Result<StagedSamplerCore<EB::Env>, Error> {
        StagedSamplerCore::build(&self.env_builder, execution_mode, obs_normalizer)
    }
}

struct Builder<E: Env> {
    env_build_plan: Box<dyn EnvBuildPlan<E>>,
    env_desription: EnvDescription<E::Tensor>,
    n_envs: usize,
    sampler_configuration: SamplerConfiguration<E>,
    backend_configuration: BackendConfiguration,
    algorithm_configuration: AlgorithmConfiguration,

    // for the hooks
    training_limit: TrainingLimit,
    learning_rate_schedule: Option<LearningRateSchedule>,
    training_artifacts: Option<TrainingArtifactsConfig>,
    control_endpoint: Option<OnPolicyControlEndpoint>,

    // for the agent
    policy_config: PolicyBuilder,
    value_hidden_layers: Vec<usize>,
    optimizer_layout: OnPolicyOptimizerLayout,
    log_progress: bool,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    gradient_clipping: Option<f32>,
    gamma: f32,
    lambda: f32,
    sample_size: usize,
    seed: Option<u64>,

    // for the sampler
    sampler_execution_mode: SamplerExecutionMode,
}

impl<E: Env> Builder<E> {
    fn new<EB: EnvBuilder<Env = E>>(
        env_builder: EB,
        n_envs: usize,
        algorithm_configuration: AlgorithmConfiguration,
        backend_configuration: BackendConfiguration,
        sampler_configuration: SamplerConfiguration<E>,
    ) -> Result<Self, Error> {
        let env_builder = EnvBuilderType::homogeneous(env_builder, n_envs)?;
        let env_desription = env_builder.env_description()?;
        Ok(Self {
            env_build_plan: Box::new(TypedEnvBuildPlan { env_builder }),
            env_desription,
            n_envs,
            sampler_configuration,
            backend_configuration,
            algorithm_configuration,
            training_limit: TrainingLimit::rollouts(300),
            learning_rate_schedule: None,
            training_artifacts: None,
            control_endpoint: None,
            policy_config: PolicyBuilder::default(),
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
            log_progress: true,
            entropy_coeff: 0.0,
            vf_coeff: None,
            gradient_clipping: None,
            gamma: 0.98,
            lambda: 0.8,
            sample_size: 64,
            seed: None,
            sampler_execution_mode: SamplerExecutionMode::MultiThreaded,
        })
    }

    fn update_optimizer_layout(
        &mut self,
        update: impl FnOnce(OnPolicyOptimizerLayout) -> OnPolicyOptimizerLayout,
    ) {
        self.optimizer_layout = update(self.optimizer_layout.clone());
    }

    fn build_candle_learner(&self, device: &Device) -> Result<CandlePolicyValueLearner, Error> {
        let observation_size = self.env_desription.observation_size();
        let (policy, policy_varmap) = self.policy_config.build_candle_with_varmap(
            observation_size,
            self.env_desription.action_space.clone(),
            device,
        )?;
        let activation_function = self.policy_config.activation_function;
        match &self.optimizer_layout {
            OnPolicyOptimizerLayout::Joint {
                max_grad_norm,
                params,
            } => CandlePolicyValueLearner::build_joint(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                *max_grad_norm,
                Self::candle_optimizer_params(params),
                activation_function,
            ),
            OnPolicyOptimizerLayout::Split {
                policy_max_grad_norm,
                policy_params,
                value_max_grad_norm,
                value_params,
            } => CandlePolicyValueLearner::build_split(
                policy,
                &self.value_hidden_layers,
                policy_varmap,
                *policy_max_grad_norm,
                *value_max_grad_norm,
                Self::candle_optimizer_params(policy_params),
                Self::candle_optimizer_params(value_params),
                activation_function,
            ),
        }
    }

    fn build_burn_learner<B: AutodiffBackend>(&self) -> Result<BurnPolicyValueLearner<B>, Error> {
        let observation_size = self.env_desription.observation_size();
        let policy = self
            .policy_config
            .build_burn::<B, _>(observation_size, self.env_desription.action_space.clone())?;
        let activation_function = self.policy_config.activation_function;
        let value_layers = &[&[observation_size][..], &self.value_hidden_layers[..], &[1]].concat();
        Ok(match &self.optimizer_layout {
            OnPolicyOptimizerLayout::Joint {
                max_grad_norm,
                params,
            } => {
                let optimizer_config = Self::burn_optimizer_config(params, *max_grad_norm);
                BurnPolicyValueLearner::joint(
                    policy,
                    value_layers,
                    activation_function,
                    &optimizer_config,
                    params.lr,
                )
            }
            OnPolicyOptimizerLayout::Split {
                policy_max_grad_norm,
                policy_params,
                value_max_grad_norm,
                value_params,
            } => {
                let optimizer_config =
                    Self::burn_optimizer_config(policy_params, *policy_max_grad_norm);
                let value_optimizer_config =
                    Self::burn_optimizer_config(value_params, *value_max_grad_norm);
                BurnPolicyValueLearner::split(
                    policy,
                    value_layers,
                    activation_function,
                    &optimizer_config,
                    policy_params.lr,
                    &value_optimizer_config,
                    value_params.lr,
                )
            }
        })
    }

    fn candle_optimizer_params(params: &AdamWParams) -> ParamsAdamW {
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

    fn default_on_policy_hook<A: Agent<Actor: ToSafetensors>, S: Sampler<Tensor = E::Tensor>>(
        self,
    ) -> Result<OnPolicyTrainingHooks<A, S, E>, Error> {
        let (evaluator, timing_recorder) = if let Some(config) = self.training_artifacts {
            let evaluator = if config.needs_evaluator() {
                let obs_normalizer = self
                    .sampler_configuration
                    .obs_normalizer()
                    .map(|n| n.with_mode(NormalizerMode::ReadOnly));
                let sampler = self.env_build_plan.build_evaluator_sampler(
                    config.evaluation_settings.episodes_per_evaluation,
                    config.evaluation_settings.evaluation_execution_mode,
                    obs_normalizer,
                )?;
                ScheduledEvaluator::new(
                    BestActorEvaluator::new(
                        sampler,
                        config.output_dir.clone(),
                        config.evaluation_results,
                        config.inference_artifacts,
                    )?,
                    config.evaluation_settings.rollouts_per_evaluation,
                )
            } else {
                ScheduledEvaluator::disabled()
            };
            let timing_recorder = if config.needs_timing_recorder() {
                TrainingTimingRecorder::create(&config.output_dir)?
            } else {
                TrainingTimingRecorder::disabled()
            };
            (evaluator, timing_recorder)
        } else {
            (
                ScheduledEvaluator::disabled(),
                TrainingTimingRecorder::disabled(),
            )
        };
        Ok(OnPolicyTrainingHooks::new(
            self.training_limit,
            self.learning_rate_schedule,
            evaluator,
            OnPolicyCommandHandler::new(self.control_endpoint),
            timing_recorder,
        ))
    }

    #[allow(clippy::unnecessary_wraps)]
    fn direct_sampler_step_bound(&self) -> Result<DirectSampler<E, StepBoundHook<E>>, Error> {
        let SamplerConfiguration::DirectStep {
            rollout_steps,
            reward_normalizer,
        } = &self.sampler_configuration
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.sampler_execution_mode);
        let step_bound_hook = StepBoundHook::new(*rollout_steps, reward_normalizer.clone());
        Ok(DirectSampler::new(sampler_core, step_bound_hook))
    }

    #[allow(clippy::unnecessary_wraps)]
    fn direct_sampler_episode_bound(&self) -> Result<DirectSampler<E, EpisodeBoundHook<E>>, Error> {
        let SamplerConfiguration::DirectEpisode { rollout_episodes } = &self.sampler_configuration
        else {
            unreachable!("direct episode-bound sampler type must use matching configuration")
        };
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.sampler_execution_mode);
        let episode_bound_hook = EpisodeBoundHook::new(*rollout_episodes);
        Ok(DirectSampler::new(sampler_core, episode_bound_hook))
    }

    fn staged_sampler_step_bound(&self) -> Result<StagedSampler<E, StepBoundHook<E>>, Error> {
        let SamplerConfiguration::StagedStep {
            rollout_steps,
            reward_normalizer,
            obs_normalizer,
        } = &self.sampler_configuration
        else {
            unreachable!("staged step-bound sampler type must use matching configuration")
        };
        let obs_normalizer = obs_normalizer
            .as_ref()
            .map(|normalizer| normalizer.with_mode(NormalizerMode::Update));
        let sampler_core = self
            .env_build_plan
            .build_staged_sampler_core(self.sampler_execution_mode, obs_normalizer)?;
        let step_bound_hook = StepBoundHook::new(*rollout_steps, reward_normalizer.clone());
        Ok(StagedSampler::new(sampler_core, step_bound_hook))
    }

    fn write_inference_config(&self, backend: InferenceBackend) -> Result<(), Error> {
        if let Some(config) = &self.training_artifacts
            && config.inference_artifacts
        {
            let observation_mode = match &self.sampler_configuration {
                SamplerConfiguration::StagedStep {
                    obs_normalizer: Some(_),
                    ..
                } => InferenceObservationMode::Normalized,
                _ => InferenceObservationMode::Raw,
            };
            let policy_builder = self.policy_config.clone();
            InferenceConfig::new(policy_builder, observation_mode, backend)
                .write_to_dir(&config.output_dir)?;
        }
        Ok(())
    }

    fn ppo_hook<M>(&mut self) -> PPOLearningHook<M> {
        let AlgorithmConfiguration::Ppo {
            normalize_advantage,
            total_epochs,
            target_kl,
            reporter,
            ..
        } = &mut self.algorithm_configuration
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        PPOLearningHook {
            normalize_advantage: normalize_advantage.unwrap_or(true),
            total_epochs: *total_epochs,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            target_kl: target_kl.map(|target| TargetKl {
                target,
                target_exceeded: false,
            }),
            gradient_clipping: self.gradient_clipping,
            current_epoch: 0,
            reporter: PPORolloutReporter::new(reporter.take(), self.log_progress, self.n_envs),
            rollout_idx: 0,
            _lm: PhantomData,
        }
    }

    fn a2c_hook<M>(&mut self) -> A2CLearningHook<M> {
        let AlgorithmConfiguration::A2C {
            normalize_advantage,
            reporter,
        } = &mut self.algorithm_configuration
        else {
            unreachable!("A2C agent type must use A2C configuration")
        };
        A2CLearningHook {
            normalize_advantage: normalize_advantage.unwrap_or(false),
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            gradient_clipping: self.gradient_clipping,
            reporter: A2CRolloutReporter::new(reporter.take(), self.log_progress, self.n_envs),
            _lm: PhantomData,
        }
    }

    fn ppo_params(&self) -> PPOParams {
        let AlgorithmConfiguration::Ppo { clip_range, .. } = &self.algorithm_configuration else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        PPOParams {
            clip_range: *clip_range,
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

    fn ppo_candle_agent(&mut self) -> Result<PPOCandle, Error> {
        let BackendConfiguration::Candle(backend) = &self.backend_configuration else {
            unreachable!("Candle agent type must use Candle backend configuration")
        };
        let backend = backend.clone();
        self.write_inference_config(InferenceBackend::Candle(backend.clone()))?;
        if let Some(seed) = self.seed {
            backend.seed(seed)?;
        }
        let learner = self.build_candle_learner(&backend.device)?;
        let hooks = self.ppo_hook();
        Ok(PPO {
            lm: learner,
            hooks,
            params: self.ppo_params(),
        })
    }

    fn ppo_burn_agent(&mut self) -> Result<PPOBurn<BurnBackend>, Error> {
        let BackendConfiguration::Burn(backend) = self.backend_configuration else {
            unreachable!("Burn agent type must use Burn backend configuration")
        };
        self.write_inference_config(InferenceBackend::Burn(backend))?;
        if let Some(seed) = self.seed {
            BurnBackend::seed(&NdArrayDevice::default(), seed);
        }
        let learner = self.build_burn_learner::<BurnBackend>()?;
        let hooks = self.ppo_hook();
        Ok(PPO {
            lm: learner,
            hooks,
            params: self.ppo_params(),
        })
    }

    fn a2c_candle_agent(&mut self) -> Result<A2CCandle, Error> {
        let BackendConfiguration::Candle(backend) = &self.backend_configuration else {
            unreachable!("Candle agent type must use Candle backend configuration")
        };
        let backend = backend.clone();
        self.write_inference_config(InferenceBackend::Candle(backend.clone()))?;
        if let Some(seed) = self.seed {
            backend.seed(seed)?;
        }
        let learner = self.build_candle_learner(&backend.device)?;
        let hooks = self.a2c_hook();
        Ok(A2C {
            lm: learner,
            hooks,
            params: self.a2c_params(),
        })
    }

    fn a2c_burn_agent(&mut self) -> Result<A2CBurn<BurnBackend>, Error> {
        let BackendConfiguration::Burn(backend) = self.backend_configuration else {
            unreachable!("Burn agent type must use Burn backend configuration")
        };
        self.write_inference_config(InferenceBackend::Burn(backend))?;
        if let Some(seed) = self.seed {
            BurnBackend::seed(&NdArrayDevice::default(), seed);
        }
        let learner = self.build_burn_learner::<BurnBackend>()?;
        let hooks = self.a2c_hook();
        Ok(A2C {
            lm: learner,
            hooks,
            params: self.a2c_params(),
        })
    }
}

struct Config<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    build_agent: fn(&mut Builder<E>) -> Result<A, Error>,
    build_sampler: fn(&Builder<E>) -> Result<S, Error>,
}

/// Configures and builds a complete PPO or A2C training algorithm.
///
/// Start with [`PPOBuilder`] or [`A2CBuilder`]. Some methods,
/// such as backend and sampler selection, return a builder with different
/// generic arguments. The concrete state components are publicly named so such
/// builders can be stored in and passed to user functions.
///
/// Both entry points default to Candle on the CPU, multi-threaded direct
/// sampling with 1,024 steps per environment, and a training limit of 300
/// rollouts. Shared learning defaults include `gamma = 0.98`, `lambda = 0.8`,
/// minibatches of 64 samples, and a joint `AdamW` optimizer with a learning rate
/// of `3e-4`.
#[must_use]
pub struct OnPolicyBuilder<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    builder: Builder<E>,
    config: Config<A, S, E>,
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>>
    OnPolicyBuilder<A, S, E>
{
    fn configured<EB: EnvBuilder<Env = E>>(
        env_builder: EB,
        n_envs: usize,
        algorithm_configuration: AlgorithmConfiguration,
        backend_configuration: BackendConfiguration,
        sampler_configuration: SamplerConfiguration<E>,
        build_agent: fn(&mut Builder<E>) -> Result<A, Error>,
        build_sampler: fn(&Builder<E>) -> Result<S, Error>,
    ) -> Result<Self, Error> {
        Ok(Self {
            builder: Builder::new(
                env_builder,
                n_envs,
                algorithm_configuration,
                backend_configuration,
                sampler_configuration,
            )?,
            config: Config {
                build_agent,
                build_sampler,
            },
        })
    }

    fn with_agent<A2: Agent>(
        self,
        build_agent: fn(&mut Builder<E>) -> Result<A2, Error>,
    ) -> OnPolicyBuilder<A2, S, E> {
        OnPolicyBuilder {
            builder: self.builder,
            config: Config {
                build_agent,
                build_sampler: self.config.build_sampler,
            },
        }
    }

    fn with_sampler<S2: Sampler<Tensor = E::Tensor>>(
        self,
        build_sampler: fn(&Builder<E>) -> Result<S2, Error>,
    ) -> OnPolicyBuilder<A, S2, E> {
        OnPolicyBuilder {
            builder: self.builder,
            config: Config {
                build_agent: self.config.build_agent,
                build_sampler,
            },
        }
    }

    /// Enables the evaluation, training-timing, and inference artifacts selected by `config`.
    ///
    /// # Arguments
    ///
    /// * `config` - The artifact output directory and the artifacts to produce.
    pub fn with_training_artifacts(mut self, config: TrainingArtifactsConfig) -> Self {
        self.builder.training_artifacts = Some(config);
        self
    }

    /// Sets the schedule that determines when training stops.
    ///
    /// # Arguments
    ///
    /// * `training_limit` - The rollout or sampled-step limit for the training run.
    pub fn with_training_limit(mut self, training_limit: TrainingLimit) -> Self {
        self.builder.training_limit = training_limit;
        self
    }

    /// Enables external control of the configured training algorithm.
    pub fn with_control(mut self) -> (Self, OnPolicyControlHandle) {
        let (control_endpoint, control_handle) = on_policy_control_channel();
        self.builder.control_endpoint = Some(control_endpoint);
        (self, control_handle)
    }

    /// Sets the learning-rate schedule applied as training progresses.
    ///
    /// Passing `None` leaves the optimizer at its configured learning rate.
    ///
    /// # Arguments
    ///
    /// * `learning_rate_schedule` - The schedule to apply, or `None` to disable scheduling.
    pub fn with_learning_rate_schedule(
        mut self,
        learning_rate_schedule: Option<LearningRateSchedule>,
    ) -> Self {
        self.builder.learning_rate_schedule = learning_rate_schedule;
        self
    }

    /// Sets the random seed used when the algorithm is built.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed used to initialize the random number generators.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.builder.seed = Some(seed);
        self
    }

    /// Selects single-threaded or multi-threaded environment execution.
    ///
    /// # Arguments
    ///
    /// * `execution_mode` - How sampler environment workers should be executed.
    pub fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.builder.sampler_execution_mode = execution_mode;
        self
    }

    /// Sets the hidden-layer widths of the policy network.
    ///
    /// # Arguments
    ///
    /// * `policy_hidden_layers` - Hidden-layer widths in network order.
    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.builder.policy_config.hidden_layers = policy_hidden_layers;
        self
    }

    /// Sets the hidden-layer activation used by the policy and value networks.
    ///
    /// # Arguments
    ///
    /// * `activation_function` - Activation function applied by hidden layers.
    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.builder.policy_config.activation_function = activation_function;
        self
    }

    /// Sets the initial log standard deviation for continuous-action policies.
    ///
    /// # Arguments
    ///
    /// * `log_std_init` - Initial logarithm of the action distribution's standard deviation.
    pub fn with_log_std_init(mut self, log_std_init: f32) -> Self {
        self.builder.policy_config.log_std_init = log_std_init;
        self
    }

    /// Sets every optimizer's learning rate and selects a constant schedule.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Constant learning rate applied to every optimizer.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_lr(learning_rate));
        self.builder.learning_rate_schedule = Some(LearningRateSchedule::Constant(learning_rate));
        self
    }

    /// Sets the `AdamW` first-moment decay for every optimizer.
    ///
    /// # Arguments
    ///
    /// * `beta1` - First-moment exponential decay coefficient.
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_beta1(beta1));
        self
    }

    /// Sets the `AdamW` second-moment decay for every optimizer.
    ///
    /// # Arguments
    ///
    /// * `beta2` - Second-moment exponential decay coefficient.
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_beta2(beta2));
        self
    }

    /// Sets the `AdamW` numerical-stability term for every optimizer.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small value added to the optimizer denominator for numerical stability.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_epsilon(epsilon));
        self
    }

    /// Sets the `AdamW` weight decay for every optimizer.
    ///
    /// # Arguments
    ///
    /// * `weight_decay` - Decoupled weight-decay coefficient.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.builder
            .update_optimizer_layout(|layout| layout.with_weight_decay(weight_decay));
        self
    }

    /// Uses one optimizer for the policy and value networks.
    ///
    /// # Arguments
    ///
    /// * `max_grad_norm` - Optional maximum gradient norm applied before each update.
    /// * `params` - AdamW parameters shared by the policy and value networks.
    pub fn with_joint(mut self, max_grad_norm: Option<f32>, params: AdamWParams) -> Self {
        self.builder
            .update_optimizer_layout(|_| OnPolicyOptimizerLayout::Joint {
                max_grad_norm,
                params,
            });
        self
    }

    /// Uses independent optimizers for the policy and value networks.
    ///
    /// # Arguments
    ///
    /// * `policy_max_grad_norm` - Optional maximum norm for policy gradients.
    /// * `policy_params` - AdamW parameters for the policy network.
    /// * `value_max_grad_norm` - Optional maximum norm for value-network gradients.
    /// * `value_params` - AdamW parameters for the value network.
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

    /// Sets the hidden-layer widths of the value network.
    ///
    /// # Arguments
    ///
    /// * `value_hidden_layers` - Hidden-layer widths in network order.
    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.builder.value_hidden_layers = value_hidden_layers;
        self
    }

    /// Enables or disables advantage normalization before learning.
    ///
    /// # Arguments
    ///
    /// * `normalize_advantage` - Whether to normalize advantages before optimizer updates.
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        match &mut self.builder.algorithm_configuration {
            AlgorithmConfiguration::Ppo {
                normalize_advantage: configured,
                ..
            }
            | AlgorithmConfiguration::A2C {
                normalize_advantage: configured,
                ..
            } => *configured = Some(normalize_advantage),
        }
        self
    }

    /// Sets the entropy term coefficient in the training loss.
    ///
    /// # Arguments
    ///
    /// * `entropy_coeff` - Multiplier applied to the entropy term.
    pub fn with_entropy_coefficient(mut self, entropy_coeff: f32) -> Self {
        self.builder.entropy_coeff = entropy_coeff;
        self
    }

    /// Sets the optional value-function loss coefficient.
    ///
    /// # Arguments
    ///
    /// * `vf_coeff` - Value-loss multiplier, or `None` to use the unscaled loss.
    pub fn with_value_loss_coefficient(mut self, vf_coeff: Option<f32>) -> Self {
        self.builder.vf_coeff = vf_coeff;
        self
    }

    /// Sets optional gradient-norm clipping in the algorithm hook.
    ///
    /// # Arguments
    ///
    /// * `gradient_clipping` - Maximum gradient norm, or `None` to disable clipping.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.builder.gradient_clipping = gradient_clipping;
        self
    }

    /// Enables or disables progress output from the learning hook.
    ///
    /// # Arguments
    ///
    /// * `log_progress` - Whether to print training progress.
    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.builder.log_progress = log_progress;
        self
    }

    /// Sets the reward discount factor.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to compute returns.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.builder.gamma = gamma;
        self
    }

    /// Sets the generalized advantage-estimation lambda.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Bias-variance tradeoff used by generalized advantage estimation.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.builder.lambda = lambda;
        self
    }

    /// Sets the minibatch size used by learning updates.
    ///
    /// # Arguments
    ///
    /// * `sample_size` - Maximum number of transitions in each learning minibatch.
    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.builder.sample_size = sample_size;
        self
    }

    /// Builds the configured agent, sampler, and training lifecycle hooks.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured training algorithm cannot be constructed.
    #[allow(clippy::type_complexity)]
    pub fn build(
        mut self,
    ) -> Result<OnPolicyAlgorithm<A, S, OnPolicyTrainingHooks<A, S, E>>, Error> {
        if let Some(seed) = self.builder.seed {
            set_seed(seed);
        }
        let agent = (self.config.build_agent)(&mut self.builder)?;
        let sampler = (self.config.build_sampler)(&self.builder)?;
        let hooks = self.builder.default_on_policy_hook()?;
        Ok(OnPolicyAlgorithm::new(
            OnPolicyRuntime { agent, sampler },
            hooks,
        ))
    }
}

impl<S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyBuilder<PPOCandle, S, E> {
    /// Uses Candle on `device` for PPO learning.
    ///
    /// # Arguments
    ///
    /// * `device` - Candle device on which policy and value learning will run.
    pub fn with_candle(mut self, device: Device) -> Self {
        self.builder.backend_configuration = BackendConfiguration::Candle(CandleBackend { device });
        self
    }

    /// Switches PPO learning to the default Burn backend.
    pub fn with_burn(mut self) -> OnPolicyBuilder<PPOBurn<BurnBackend>, S, E> {
        self.builder.backend_configuration = BackendConfiguration::Burn(BurnBackendConfig);
        self.with_agent(Builder::ppo_burn_agent)
    }
}

impl<S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyBuilder<PPOBurn<BurnBackend>, S, E> {
    /// Switches PPO learning to Candle on `device`.
    ///
    /// # Arguments
    ///
    /// * `device` - Candle device on which policy and value learning will run.
    pub fn with_candle(mut self, device: Device) -> OnPolicyBuilder<PPOCandle, S, E> {
        self.builder.backend_configuration = BackendConfiguration::Candle(CandleBackend { device });
        self.with_agent(Builder::ppo_candle_agent)
    }

    /// Keeps PPO learning on the default Burn backend.
    pub fn with_burn(mut self) -> Self {
        self.builder.backend_configuration = BackendConfiguration::Burn(BurnBackendConfig);
        self
    }
}

impl<M, S, E> OnPolicyBuilder<PPO<M, PPOLearningHook<M>>, S, E>
where
    M: OnPolicyLearner,
    M::InferencePolicy: ToSafetensors,
    PPOLearningHook<M>: PPOHook<M>,
    S: Sampler,
    E: Env<Tensor = S::Tensor>,
{
    /// Installs an optional channel for reporting PPO training statistics.
    ///
    /// # Arguments
    ///
    /// * `tx` - Channel sender that receives rollout statistics, or `None` to disable reporting.
    pub fn with_rollout_reporter(mut self, tx: Option<Sender<PPORolloutStats>>) -> Self {
        let AlgorithmConfiguration::Ppo { reporter, .. } =
            &mut self.builder.algorithm_configuration
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        *reporter = tx;
        self
    }

    /// Sets the maximum PPO epochs performed for each rollout.
    ///
    /// # Arguments
    ///
    /// * `total_epochs` - Maximum optimization epochs performed over each rollout.
    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        let AlgorithmConfiguration::Ppo {
            total_epochs: configured,
            ..
        } = &mut self.builder.algorithm_configuration
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        *configured = total_epochs;
        self
    }

    /// Sets the optional KL-divergence threshold for stopping PPO epochs early.
    ///
    /// # Arguments
    ///
    /// * `target_kl` - KL-divergence threshold, or `None` to disable early stopping.
    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        let AlgorithmConfiguration::Ppo {
            target_kl: configured,
            ..
        } = &mut self.builder.algorithm_configuration
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        *configured = target_kl;
        self
    }

    /// Sets the PPO policy-ratio clipping range.
    ///
    /// # Arguments
    ///
    /// * `clip_range` - Maximum allowed deviation of the policy ratio from `1.0`.
    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        let AlgorithmConfiguration::Ppo {
            clip_range: configured,
            ..
        } = &mut self.builder.algorithm_configuration
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        *configured = clip_range;
        self
    }
}

impl<S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyBuilder<A2CCandle, S, E> {
    /// Uses Candle on `device` for A2C learning.
    ///
    /// # Arguments
    ///
    /// * `device` - Candle device on which policy and value learning will run.
    pub fn with_candle(mut self, device: Device) -> Self {
        self.builder.backend_configuration = BackendConfiguration::Candle(CandleBackend { device });
        self
    }

    /// Switches A2C learning to the default Burn backend.
    pub fn with_burn(mut self) -> OnPolicyBuilder<A2CBurn<BurnBackend>, S, E> {
        self.builder.backend_configuration = BackendConfiguration::Burn(BurnBackendConfig);
        self.with_agent(Builder::a2c_burn_agent)
    }
}

impl<S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyBuilder<A2CBurn<BurnBackend>, S, E> {
    /// Switches A2C learning to Candle on `device`.
    ///
    /// # Arguments
    ///
    /// * `device` - Candle device on which policy and value learning will run.
    pub fn with_candle(mut self, device: Device) -> OnPolicyBuilder<A2CCandle, S, E> {
        self.builder.backend_configuration = BackendConfiguration::Candle(CandleBackend { device });
        self.with_agent(Builder::a2c_candle_agent)
    }

    /// Keeps A2C learning on the default Burn backend.
    pub fn with_burn(mut self) -> Self {
        self.builder.backend_configuration = BackendConfiguration::Burn(BurnBackendConfig);
        self
    }
}

impl<M, S, E> OnPolicyBuilder<A2C<M, A2CLearningHook<M>>, S, E>
where
    M: OnPolicyLearner,
    M::InferencePolicy: ToSafetensors,
    A2CLearningHook<M>: A2CHook<M>,
    S: Sampler,
    E: Env<Tensor = S::Tensor>,
{
    /// Installs an optional channel for reporting A2C training statistics.
    ///
    /// # Arguments
    ///
    /// * `tx` - Channel sender that receives rollout statistics, or `None` to disable reporting.
    pub fn with_rollout_reporter(mut self, tx: Option<Sender<A2CRolloutStats>>) -> Self {
        let AlgorithmConfiguration::A2C { reporter, .. } =
            &mut self.builder.algorithm_configuration
        else {
            unreachable!("A2C agent type must use A2C configuration")
        };
        *reporter = tx;
        self
    }
}

impl<A: Agent<Actor: ToSafetensors>, E: Env>
    OnPolicyBuilder<A, DirectSampler<E, StepBoundHook<E>>, E>
{
    /// Sets the number of steps collected per environment and rollout.
    ///
    /// # Arguments
    ///
    /// * `rollout_steps` - Number of steps collected from each environment per rollout.
    pub fn with_rollout_steps(mut self, rollout_steps: usize) -> Self {
        let SamplerConfiguration::DirectStep {
            rollout_steps: configured,
            ..
        } = &mut self.builder.sampler_configuration
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        *configured = rollout_steps;
        self
    }

    /// Normalizes discounted rewards and clips them to `clip_reward`.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to track discounted returns.
    /// * `clip_reward` - Absolute limit applied to normalized rewards.
    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        let SamplerConfiguration::DirectStep {
            reward_normalizer, ..
        } = &mut self.builder.sampler_configuration
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        *reward_normalizer = Some(RewardNormalizer::new(
            self.builder.n_envs,
            gamma,
            clip_reward,
        ));
        self
    }

    /// Selects staged sampling and optionally enables clipped observation normalization.
    ///
    /// `Some(clip)` enables normalization with that clipping limit. `None`
    /// retains staged sampling without applying an observation normalizer.
    ///
    /// # Arguments
    ///
    /// * `obs_clip` - Absolute clipping limit, or `None` to keep observations unnormalized.
    ///
    /// # Errors
    ///
    /// Returns an error if an observation normalizer cannot be constructed.
    #[allow(clippy::type_complexity)]
    pub fn with_observation_normalizer(
        mut self,
        obs_clip: Option<f32>,
    ) -> Result<OnPolicyBuilder<A, StagedSampler<E, StepBoundHook<E>>, E>, Error> {
        let obs_normalizer = obs_clip
            .map(|clip| {
                ClippedNormalizer::build(
                    NormalizerMode::Update,
                    clip,
                    vec![self.builder.env_desription.observation_space.size()],
                )
            })
            .transpose()?;
        let SamplerConfiguration::DirectStep {
            rollout_steps,
            reward_normalizer,
        } = self.builder.sampler_configuration
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        self.builder.sampler_configuration = SamplerConfiguration::StagedStep {
            rollout_steps,
            reward_normalizer,
            obs_normalizer,
        };
        Ok(self.with_sampler(Builder::staged_sampler_step_bound))
    }

    /// Selects direct sampling bounded by completed episodes per environment.
    ///
    /// # Arguments
    ///
    /// * `rollout_episodes` - Number of completed episodes collected from each environment per
    ///   rollout.
    pub fn with_rollout_episodes(
        mut self,
        rollout_episodes: usize,
    ) -> OnPolicyBuilder<A, DirectSampler<E, EpisodeBoundHook<E>>, E> {
        let SamplerConfiguration::DirectStep { .. } = self.builder.sampler_configuration else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        self.builder.sampler_configuration =
            SamplerConfiguration::DirectEpisode { rollout_episodes };
        self.with_sampler(Builder::direct_sampler_episode_bound)
    }
}

impl<A: Agent<Actor: ToSafetensors>, E: Env>
    OnPolicyBuilder<A, StagedSampler<E, StepBoundHook<E>>, E>
{
    /// Sets the number of steps collected per environment and rollout.
    ///
    /// # Arguments
    ///
    /// * `rollout_steps` - Number of steps collected from each environment per rollout.
    pub fn with_rollout_steps(mut self, rollout_steps: usize) -> Self {
        let SamplerConfiguration::StagedStep {
            rollout_steps: configured,
            ..
        } = &mut self.builder.sampler_configuration
        else {
            unreachable!("staged step-bound sampler type must use matching configuration")
        };
        *configured = rollout_steps;
        self
    }

    /// Normalizes discounted rewards and clips them to `clip_reward`.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to track discounted returns.
    /// * `clip_reward` - Absolute limit applied to normalized rewards.
    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        let SamplerConfiguration::StagedStep {
            reward_normalizer, ..
        } = &mut self.builder.sampler_configuration
        else {
            unreachable!("staged step-bound sampler type must use matching configuration")
        };
        *reward_normalizer = Some(RewardNormalizer::new(
            self.builder.n_envs,
            gamma,
            clip_reward,
        ));
        self
    }
}

/// Default PPO builder using Candle and direct, step-bounded sampling.
pub type PPOBuilder<E> = OnPolicyBuilder<PPOCandle, DirectSampler<E, StepBoundHook<E>>, E>;

/// Default A2C builder using Candle and direct, step-bounded sampling.
pub type A2CBuilder<E> = OnPolicyBuilder<A2CCandle, DirectSampler<E, StepBoundHook<E>>, E>;

impl<E: Env> PPOBuilder<E> {
    /// Creates a PPO builder using homogeneous environments.
    ///
    /// # Arguments
    ///
    /// * `env_builder` - The environment builder used to create each environment instance.
    /// * `num_envs` - The number of independent environment instances used to collect rollouts.
    ///   Each environment collects the configured number of rollout steps, so the default 1,024
    ///   steps with four environments produces 4,096 transitions per rollout.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_envs` is zero.
    pub fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, num_envs: usize) -> Result<Self, Error> {
        Self::configured(
            env_builder,
            num_envs,
            AlgorithmConfiguration::Ppo {
                normalize_advantage: None,
                total_epochs: 10,
                target_kl: None,
                clip_range: 0.2,
                reporter: None,
            },
            BackendConfiguration::Candle(CandleBackend {
                device: Device::Cpu,
            }),
            SamplerConfiguration::DirectStep {
                rollout_steps: 1024,
                reward_normalizer: None,
            },
            Builder::ppo_candle_agent,
            Builder::direct_sampler_step_bound,
        )
    }
}

impl<E: Env> A2CBuilder<E> {
    /// Creates an A2C builder using homogeneous environments.
    ///
    /// # Arguments
    ///
    /// * `env_builder` - The environment builder used to create each environment instance.
    /// * `num_envs` - The number of independent environment instances used to collect rollouts.
    ///   Each environment collects the configured number of rollout steps, so the default 1,024
    ///   steps with four environments produces 4,096 transitions per rollout.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_envs` is zero.
    pub fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, num_envs: usize) -> Result<Self, Error> {
        Self::configured(
            env_builder,
            num_envs,
            AlgorithmConfiguration::A2C {
                normalize_advantage: None,
                reporter: None,
            },
            BackendConfiguration::Candle(CandleBackend {
                device: Device::Cpu,
            }),
            SamplerConfiguration::DirectStep {
                rollout_steps: 1024,
                reward_normalizer: None,
            },
            Builder::a2c_candle_agent,
            Builder::direct_sampler_step_bound,
        )
    }
}

impl PPOBuilder<GymEnv> {
    /// Creates a PPO builder for a Gymnasium environment.
    ///
    /// # Arguments
    ///
    /// * `env_builder` - The Gymnasium environment name or configured [`GymEnvBuilder`] used to
    ///   create each environment instance.
    /// * `num_envs` - The number of independent environment instances used to collect rollouts.
    ///   Each environment collects the configured number of rollout steps, so the default 1,024
    ///   steps with four environments produces 4,096 transitions per rollout.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_envs` is zero.
    pub fn gym<EB: Into<GymEnvBuilder>>(env_builder: EB, num_envs: usize) -> Result<Self, Error> {
        Self::new(env_builder.into(), num_envs)
    }
}

impl A2CBuilder<GymEnv> {
    /// Creates an A2C builder for a Gymnasium environment.
    ///
    /// # Arguments
    ///
    /// * `env_builder` - The Gymnasium environment name or configured [`GymEnvBuilder`] used to
    ///   create each environment instance.
    /// * `num_envs` - The number of independent environment instances used to collect rollouts.
    ///   Each environment collects the configured number of rollout steps, so the default 1,024
    ///   steps with four environments produces 4,096 transitions per rollout.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_envs` is zero.
    pub fn gym<EB: Into<GymEnvBuilder>>(env_builder: EB, num_envs: usize) -> Result<Self, Error> {
        Self::new(env_builder.into(), num_envs)
    }
}
