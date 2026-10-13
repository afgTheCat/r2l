pub(crate) mod algorithm;
/// Learner and learning-hook configuration.
pub mod learner;
/// Backend-independent network configuration and construction.
pub mod networks;
/// Training lifecycle and artifact configuration.
pub mod on_policy_hook;
/// Adam optimizer settings, learning-rate schedules, and clipping.
pub mod optimizer;
pub(crate) mod policy;
pub(crate) mod sampler;

use std::{num::NonZeroUsize, sync::mpsc::Sender};

pub use algorithm::{A2CConfig, AlgoBackendConfig, PPOConfig};
use networks::NetworkConfig;
pub use on_policy_hook::TrainingArtifactsConfig;
pub use optimizer::{AdamWConfig, GradientClippingConfig, LearningRateSchedule, OptimizerConfig};
use r2l_agents::on_policy_algorithms::{a2c::A2C, ppo::PPO};
use r2l_core::error::Result;
use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    models::{ActivationFunction, ToSafetensors},
    on_policy::algorithm::{Agent, OnPolicyAlgorithm, OnPolicyRuntime},
};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearner as BurnPolicyValueLearner;
use r2l_distributions::learning_modules::candle_lm::PolicyValueLearner as CandlePolicyValueLearner;
#[cfg(feature = "gym")]
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{RolloutMode, SamplerExecutionMode};
pub use sampler::ObsNormalizerConfig;
use serde::{Deserialize, Serialize};

pub use crate::backend::{BurnBackendConfig, CandleDevice};
use crate::{
    A2CRolloutStats, OnPolicyControlEndpoint, OnPolicyHooks, PPORolloutStats, TrainingLimit,
    backend::{Backend, BackendConfig},
    builders::{
        algorithm::AlgoConfig,
        learner::{LearnerConfig, LearningHookConfig},
        on_policy_hook::OnPolicyHookConfig,
        sampler::{RewardNormalizerConfig, SamplerConfiguration},
    },
    hooks::{
        learning::{A2CLearningHook, ClipRangeSchedule, PPOLearningHook},
        progress::{SharedTrainingProgress, TrainingProgress},
    },
};

/// PPO agent produced by a Candle-backed algorithm builder.
pub type PPOCandle = PPO<CandlePolicyValueLearner, PPOLearningHook>;
/// PPO agent produced by a Burn-backed algorithm builder.
pub type PPOBurn = PPO<BurnPolicyValueLearner, PPOLearningHook>;
/// A2C agent produced by a Candle-backed algorithm builder.
pub type A2CCandle = A2C<CandlePolicyValueLearner, A2CLearningHook>;
/// A2C agent produced by a Burn-backed algorithm builder.
pub type A2CBurn = A2C<BurnPolicyValueLearner, A2CLearningHook>;

#[derive(Serialize, Deserialize)]
pub(crate) struct BuilderConfig {
    algo_config: AlgoConfig,
    seed: Option<u64>,
    hook_config: OnPolicyHookConfig,
    learner_builder: LearnerConfig,
    learning_hook: LearningHookConfig,
    sampler_configuration: SamplerConfiguration,
}

pub(crate) struct BuilderResources<E: Env + 'static> {
    pub(crate) env_builder: EnvBuilderType<E>,
    pub(crate) control_endpoint: Option<OnPolicyControlEndpoint>,
    pub(crate) ppo_reporter: Option<Sender<PPORolloutStats>>,
    pub(crate) a2c_reporter: Option<Sender<A2CRolloutStats>>,
}

impl<E: Env> BuilderResources<E> {
    pub(crate) fn n_envs(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.env_builder.num_envs()).expect("environment builders are nonempty")
    }
}

/// Configures and builds a complete PPO or A2C training algorithm.
///
/// Start with [`PPOBuilder`] or [`A2CBuilder`]. `A` and `B` identify the algorithm
/// and backend configurations. Backend selection changes the backend type
/// while retaining all algorithm settings and runtime resources.
///
/// Both entry points default to Candle on the CPU, multi-threaded direct
/// sampling with 1,024 steps per environment, and a training limit of 50
/// rollouts. Shared learning defaults include `gamma = 0.98`, `lambda = 0.8`,
/// minibatches of 64 samples, and a joint Adam optimizer with decoupled weight
/// decay and a learning rate of `3e-4`.
#[must_use]
pub struct OnPolicyBuilder<A, B, E: Env + 'static> {
    algo_backend: AlgoBackendConfig<A, B>,
    config: BuilderConfig,
    resources: BuilderResources<E>,
}

impl<A, B, E: Env> OnPolicyBuilder<A, B, E> {
    fn configured<EB: EnvBuilder<Env = E>>(
        env_builder: EB,
        n_envs: usize,
        algo_backend: AlgoBackendConfig<A, B>,
    ) -> Result<Self> {
        let env_builder = EnvBuilderType::homogeneous(env_builder, n_envs)?;
        let learner_builder = LearnerConfig::new(&env_builder.env_description()?)?;
        Ok(Self {
            algo_backend,
            config: BuilderConfig {
                algo_config: AlgoConfig::default(),
                seed: None,
                hook_config: OnPolicyHookConfig::default(),
                learner_builder,
                learning_hook: LearningHookConfig::new(true),
                sampler_configuration: SamplerConfiguration::default(),
            },
            resources: BuilderResources {
                env_builder,
                control_endpoint: None,
                ppo_reporter: None,
                a2c_reporter: None,
            },
        })
    }

    fn progress(&self) -> SharedTrainingProgress {
        TrainingProgress::shared(
            self.config.hook_config.training_limit,
            self.config.sampler_configuration.rollout_mode,
            self.resources.n_envs(),
        )
    }

    /// Enables the evaluation, training-timing, and inference artifacts selected by `config`.
    ///
    /// # Arguments
    ///
    /// * `config` - The artifact output directory and the artifacts to produce.
    pub fn with_training_artifacts(mut self, config: TrainingArtifactsConfig) -> Self {
        self.config.hook_config.training_artifacts = Some(config);
        self
    }

    /// Sets the schedule that determines when training stops.
    ///
    /// # Arguments
    ///
    /// * `training_limit` - The rollout or sampled-step limit for the training run.
    pub fn with_training_limit(mut self, training_limit: TrainingLimit) -> Self {
        self.config.hook_config.training_limit = training_limit;
        self
    }

    /// Stops training when an evaluation's average episode reward reaches the threshold.
    ///
    /// Enabling this condition also enables evaluation without requiring training artifacts.
    /// By default, evaluation uses modal actions for five episodes per environment after every
    /// training rollout. If training artifacts are configured, their evaluation settings apply.
    /// The configured training limit still applies when the threshold has not been reached.
    ///
    /// # Arguments
    ///
    /// * `avg_reward_threshold` - Stop when the mean episode reward is at least this value, or
    ///   `None` to disable reward-based stopping. Evaluation uses unnormalized environment rewards.
    ///
    /// # Panics
    ///
    /// Panics if a supplied threshold is not finite.
    pub fn with_avg_reward_threshold(mut self, avg_reward_threshold: Option<f32>) -> Self {
        assert!(
            avg_reward_threshold.is_none_or(f32::is_finite),
            "average reward threshold must be finite"
        );
        self.config.hook_config.avg_reward_threshold = avg_reward_threshold;
        self
    }

    /// Enables external control of the configured training algorithm.
    ///
    /// # Arguments
    ///
    /// * `control_endpoint` - Algorithm-side endpoint created by
    ///   [`OnPolicyControlEndpoint::channel`]. Replaces any previously configured endpoint.
    pub fn with_control(mut self, control_endpoint: OnPolicyControlEndpoint) -> Self {
        self.resources.control_endpoint = Some(control_endpoint);
        self
    }

    /// Sets the learning-rate schedule applied as training progresses.
    ///
    /// Defaults to `Constant(3e-4)`. Updates every optimizer configuration, including
    /// both policy and value configurations in split mode.
    ///
    /// # Arguments
    ///
    /// * `learning_rate_schedule` - Schedule applied to every optimizer before learning.
    pub fn with_learning_rate_schedule(
        mut self,
        learning_rate_schedule: LearningRateSchedule,
    ) -> Self {
        self.config
            .learner_builder
            .optimizer
            .for_each_mut(|config| {
                config.learning_rate = learning_rate_schedule;
            });
        self
    }

    /// Sets the random seed used when the algorithm is built.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed used to initialize the random number generators.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Selects single-threaded or multi-threaded environment execution.
    ///
    /// # Arguments
    ///
    /// * `execution_mode` - How sampler environment workers should be executed.
    pub fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.config.sampler_configuration.execution_mode = execution_mode;
        self
    }

    /// Selects the policy architecture. CNN construction currently requires Burn.
    /// CNNs reshape flat observations using the environment's `[channels, height, width]` shape.
    ///
    /// # Arguments
    ///
    /// * `network` - Hidden architecture; observation shape and action-space outputs are inferred.
    pub fn with_policy_network(mut self, network: NetworkConfig) -> Self {
        self.config.learner_builder.policy_config.network = network;
        self
    }

    /// Selects the value architecture independently of the policy. CNNs currently require Burn.
    /// CNNs reshape flat observations using the environment's `[channels, height, width]` shape.
    ///
    /// # Arguments
    ///
    /// * `network` - Hidden architecture; observation shape is inferred and output width is one.
    pub fn with_value_network(mut self, network: NetworkConfig) -> Self {
        self.config.learner_builder.value_network = network;
        self
    }

    /// Sets the hidden-layer widths of the policy network.
    ///
    /// # Arguments
    ///
    /// * `policy_hidden_layers` - Dense hidden-layer widths, including the dense layers after a CNN.
    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.config
            .learner_builder
            .policy_config
            .network
            .mlp_config_mut()
            .hidden_layers = policy_hidden_layers;
        self
    }

    /// Sets dense hidden-layer activation for both networks, preserving explicit CNN activations.
    ///
    /// # Arguments
    ///
    /// * `activation_function` - Activation function applied by hidden layers.
    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.config
            .learner_builder
            .policy_config
            .network
            .mlp_config_mut()
            .activation = activation_function;
        self.config
            .learner_builder
            .value_network
            .mlp_config_mut()
            .activation = activation_function;
        self
    }

    /// Sets the initial log standard deviation for continuous-action policies.
    ///
    /// # Arguments
    ///
    /// * `log_std_init` - Initial logarithm of the action distribution's standard deviation.
    pub fn with_log_std_init(mut self, log_std_init: f32) -> Self {
        self.config.learner_builder.policy_config.log_std_init = log_std_init;
        self
    }

    /// Enables generalized state-dependent exploration for continuous Box actions.
    ///
    /// Each environment gets independent noise at each rollout boundary. Evaluation
    /// uses the policy mean. With gSDE, `log_std_init` initializes the noise weights
    /// for each hidden feature and action, rather than the action's marginal scale.
    ///
    /// # Arguments
    ///
    /// * `config` - Controls additional noise refreshes within a rollout.
    pub fn with_sde(mut self, config: crate::SdeConfig) -> Self {
        self.config.learner_builder.policy_config.sde = Some(config);
        self
    }

    /// Sets every optimizer's learning rate and selects a constant schedule.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Constant learning rate applied to every optimizer.
    pub fn with_learning_rate(self, learning_rate: f64) -> Self {
        self.with_learning_rate_schedule(LearningRateSchedule::Constant(learning_rate))
    }

    /// Sets the Adam first-moment decay for every optimizer.
    ///
    /// # Arguments
    ///
    /// * `beta1` - First-moment exponential decay coefficient.
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.config
            .learner_builder
            .optimizer
            .for_each_mut(|config| config.beta1 = beta1);
        self
    }

    /// Sets the Adam second-moment decay for every optimizer.
    ///
    /// # Arguments
    ///
    /// * `beta2` - Second-moment exponential decay coefficient.
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.config
            .learner_builder
            .optimizer
            .for_each_mut(|config| config.beta2 = beta2);
        self
    }

    /// Sets the Adam numerical-stability term for every optimizer.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small value added to the optimizer denominator for numerical stability.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.config
            .learner_builder
            .optimizer
            .for_each_mut(|config| config.eps = epsilon);
        self
    }

    /// Sets the decoupled weight decay for every Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `weight_decay` - Decoupled weight-decay coefficient.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.config
            .learner_builder
            .optimizer
            .for_each_mut(|config| config.weight_decay = weight_decay);
        self
    }

    /// Replaces the complete optimizer configuration.
    ///
    /// Later optimizer setters update this configuration. In split mode, shared
    /// setters update both optimizers; use `OptimizerConfig::Split` for independent settings.
    ///
    /// # Arguments
    ///
    /// * `config` - Joint or split Adam optimizers, including schedules and gradient clipping.
    pub fn with_optimizer(mut self, config: OptimizerConfig) -> Self {
        self.config.learner_builder.optimizer = config;
        self
    }

    /// Sets the hidden-layer widths of the value network.
    ///
    /// # Arguments
    ///
    /// * `value_hidden_layers` - Dense hidden-layer widths, including the dense layers after a CNN.
    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.config
            .learner_builder
            .value_network
            .mlp_config_mut()
            .hidden_layers = value_hidden_layers;
        self
    }

    /// Enables or disables advantage normalization before learning.
    ///
    /// # Arguments
    ///
    /// * `normalize_advantage` - Whether to normalize advantages before optimizer updates.
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.config.learning_hook.normalize_advantage = normalize_advantage;
        self
    }

    /// Sets the entropy term coefficient in the training loss.
    ///
    /// # Arguments
    ///
    /// * `entropy_coeff` - Multiplier applied to the entropy term.
    pub fn with_entropy_coefficient(mut self, entropy_coeff: f32) -> Self {
        self.config.learning_hook.entropy_coeff = entropy_coeff;
        self
    }

    /// Sets the value-function loss coefficient, which defaults to `1.0`.
    ///
    /// # Arguments
    ///
    /// * `vf_coeff` - Value-loss multiplier; `1.0` uses the unscaled loss.
    pub fn with_value_loss_coefficient(mut self, vf_coeff: f32) -> Self {
        self.config.learning_hook.vf_coeff = vf_coeff;
        self
    }

    /// Sets gradient clipping for every optimizer.
    ///
    /// Clipping is configured at build time and applied to gradients during updates.
    /// In split mode, both optimizers receive this setting and clip independently.
    /// The default is [`GradientClippingConfig::Disabled`].
    ///
    /// # Arguments
    ///
    /// * `gradient_clipping` - Clipping mode; `Disabled` clears the optimizer's clipping.
    pub fn with_gradient_clipping(mut self, gradient_clipping: GradientClippingConfig) -> Self {
        self.config
            .learner_builder
            .optimizer
            .for_each_mut(|config| {
                config.gradient_clipping = gradient_clipping;
            });
        self
    }

    /// Enables or disables progress output from the learning hook.
    ///
    /// # Arguments
    ///
    /// * `log_progress` - Whether to print training progress.
    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.config.algo_config.log_progress = log_progress;
        self
    }

    /// Sets the reward discount factor.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to compute returns.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.config.algo_config.gamma = gamma;
        self
    }

    /// Sets the generalized advantage-estimation lambda.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Bias-variance tradeoff used by generalized advantage estimation.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.config.algo_config.lambda = lambda;
        self
    }

    /// Sets the minibatch size used by learning updates.
    ///
    /// # Arguments
    ///
    /// * `sample_size` - Maximum number of transitions in each learning minibatch.
    ///
    /// # Panics
    ///
    /// Panics if `sample_size` is zero.
    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        let sample_size =
            NonZeroUsize::new(sample_size).expect("sample size must be greater than zero");
        self.config.algo_config.sample_size = sample_size;
        self
    }

    fn finish_build<AG: Agent<Actor: ToSafetensors>>(
        self,
        agent: AG,
        backend: BackendConfig,
        progress: SharedTrainingProgress,
    ) -> Result<OnPolicyAlgorithm<OnPolicyHooks<AG, E>>> {
        let policy = &self.config.learner_builder.policy_config;
        let obs_normalizer = self
            .config
            .sampler_configuration
            .build_obs_normalizer(policy.observation_space.observation_shape());
        let sampler = self
            .config
            .sampler_configuration
            .build_sampler(&self.resources.env_builder, obs_normalizer.clone())?;
        let hooks = self.config.hook_config.build(
            self.resources,
            policy,
            backend,
            obs_normalizer,
            progress,
        )?;
        Ok(OnPolicyAlgorithm::new(
            OnPolicyRuntime::new(agent, sampler),
            hooks,
        ))
    }

    /// Uses Candle on `device` while preserving the algorithm settings.
    ///
    /// # Arguments
    ///
    /// * `device` - Device kind and ordinal to initialize when building the algorithm.
    pub fn with_candle(self, device: CandleDevice) -> OnPolicyBuilder<A, CandleDevice, E> {
        self.with_backend(device)
    }

    /// Uses the default Burn backend while preserving the algorithm settings.
    pub fn with_burn(self) -> OnPolicyBuilder<A, BurnBackendConfig, E> {
        self.with_backend(BurnBackendConfig)
    }

    fn with_backend<B2>(self, backend: B2) -> OnPolicyBuilder<A, B2, E> {
        OnPolicyBuilder {
            algo_backend: AlgoBackendConfig {
                algo: self.algo_backend.algo,
                backend,
            },
            config: self.config,
            resources: self.resources,
        }
    }

    /// Sets the number of steps collected per environment and rollout.
    ///
    /// # Arguments
    ///
    /// * `rollout_steps` - Number of steps collected from each environment per rollout.
    ///
    /// # Panics
    ///
    /// Panics if `rollout_steps` is zero.
    pub fn with_rollout_steps(mut self, rollout_steps: usize) -> Self {
        self.config.sampler_configuration.rollout_mode = RolloutMode::StepBound {
            n_steps: NonZeroUsize::new(rollout_steps)
                .expect("rollout steps must be greater than zero"),
        };
        self
    }

    /// Normalizes rewards using discounted returns and clips them to `clip_reward`.
    /// Works with both step-bound and episode-bound collection.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to track discounted returns.
    /// * `clip_reward` - Absolute limit applied to normalized rewards.
    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        self.config.sampler_configuration.reward_normalizer =
            Some(RewardNormalizerConfig { gamma, clip_reward });
        self
    }

    /// Configures observation normalization and selects the collection strategy.
    ///
    /// Enabled normalization can optionally clip the normalized values. Disabled
    /// normalization selects direct collection without a normalizer.
    ///
    /// # Arguments
    ///
    /// * `config` - Whether to normalize observations and the optional absolute clipping limit.
    pub fn with_observation_normalizer(mut self, config: ObsNormalizerConfig) -> Self {
        self.config.sampler_configuration.obs_normalizer = config;
        self
    }

    /// Collects complete episodes per environment, retaining observation and reward normalization.
    ///
    /// # Arguments
    ///
    /// * `rollout_episodes` - Number of completed episodes collected from each environment per
    ///   rollout.
    ///
    /// # Panics
    ///
    /// Panics if `rollout_episodes` is zero.
    pub fn with_rollout_episodes(mut self, rollout_episodes: usize) -> Self {
        self.config.sampler_configuration.rollout_mode = RolloutMode::EpisodeBound {
            n_episodes: NonZeroUsize::new(rollout_episodes)
                .expect("rollout episodes must be greater than zero"),
        };
        self
    }
}

impl<A, E: Env> OnPolicyBuilder<A, CandleDevice, E> {
    fn build_learner(&self) -> Result<CandlePolicyValueLearner> {
        let device = self.algo_backend.backend.build()?;
        if let Some(seed) = self.config.seed {
            Backend::Candle(device.clone()).seed(seed)?;
        }
        self.config.learner_builder.build_candle_learner(&device)
    }
}

impl<A, E: Env> OnPolicyBuilder<A, BurnBackendConfig, E> {
    fn build_learner(&self) -> Result<BurnPolicyValueLearner> {
        if let Some(seed) = self.config.seed {
            Backend::Burn.seed(seed)?;
        }
        self.config.learner_builder.build_burn_learner()
    }
}

impl<B, E: Env> OnPolicyBuilder<PPOConfig, B, E> {
    /// Installs an optional channel for reporting PPO training statistics.
    ///
    /// # Arguments
    ///
    /// * `tx` - Channel sender that receives rollout statistics, or `None` to disable reporting.
    pub fn with_rollout_reporter(mut self, tx: Option<Sender<PPORolloutStats>>) -> Self {
        self.resources.ppo_reporter = tx;
        self
    }

    /// Sets the maximum PPO epochs performed for each rollout.
    ///
    /// # Arguments
    ///
    /// * `total_epochs` - Maximum optimization epochs performed over each rollout.
    ///
    /// # Panics
    ///
    /// Panics if `total_epochs` is zero.
    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.algo_backend.algo.total_epochs =
            NonZeroUsize::new(total_epochs).expect("total epochs must be greater than zero");
        self
    }

    /// Sets the optional KL-divergence threshold for stopping PPO epochs early.
    ///
    /// # Arguments
    ///
    /// * `target_kl` - KL-divergence threshold, or `None` to disable early stopping.
    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.algo_backend.algo.target_kl = target_kl;
        self
    }

    /// Sets the PPO policy-ratio clipping range.
    ///
    /// # Arguments
    ///
    /// * `clip_range` - Maximum allowed deviation of the policy ratio from `1.0`.
    pub fn with_clip_range(self, clip_range: f32) -> Self {
        self.with_clip_range_schedule(ClipRangeSchedule::Constant(clip_range))
    }

    /// Sets the PPO policy-ratio clipping schedule.
    ///
    /// # Arguments
    ///
    /// * `clip_range_schedule` - Schedule applied to the clipping range as training progresses.
    pub fn with_clip_range_schedule(mut self, clip_range_schedule: ClipRangeSchedule) -> Self {
        self.algo_backend.algo.clip_range_schedule = clip_range_schedule;
        self
    }
}

impl<B, E: Env> OnPolicyBuilder<A2CConfig, B, E> {
    /// Installs an optional channel for reporting A2C training statistics.
    ///
    /// # Arguments
    ///
    /// * `tx` - Channel sender that receives rollout statistics, or `None` to disable reporting.
    pub fn with_rollout_reporter(mut self, tx: Option<Sender<A2CRolloutStats>>) -> Self {
        self.resources.a2c_reporter = tx;
        self
    }
}

impl<E: Env> OnPolicyBuilder<PPOConfig, CandleDevice, E> {
    /// Builds the configured Candle PPO training algorithm.
    ///
    /// Move the builder to the training thread before building; the algorithm is not `Send`.
    ///
    /// # Errors
    ///
    /// Returns an error if the learner, sampler, or lifecycle hooks cannot be constructed.
    ///
    /// # Panics
    ///
    /// Panics if the tensor backend cannot create enabled observation-normalization statistics.
    pub fn build(mut self) -> Result<OnPolicyAlgorithm<OnPolicyHooks<PPOCandle, E>>> {
        let progress = self.progress();
        let learner = self.build_learner()?;
        let agent = self.config.algo_config.build_ppo(
            &self.algo_backend.algo,
            learner,
            &self.config.learning_hook,
            &self.config.learner_builder.optimizer,
            progress.clone(),
            self.resources.ppo_reporter.take(),
            self.resources.n_envs(),
        );
        let backend = BackendConfig::Candle(self.algo_backend.backend);
        self.finish_build(agent, backend, progress)
    }
}

impl<E: Env> OnPolicyBuilder<PPOConfig, BurnBackendConfig, E> {
    /// Builds the configured Burn PPO training algorithm.
    ///
    /// Move the builder to the training thread before building; the algorithm is not `Send`.
    ///
    /// # Errors
    ///
    /// Returns an error if the learner, sampler, or lifecycle hooks cannot be constructed.
    ///
    /// # Panics
    ///
    /// Panics if the tensor backend cannot create enabled observation-normalization statistics.
    pub fn build(mut self) -> Result<OnPolicyAlgorithm<OnPolicyHooks<PPOBurn, E>>> {
        let progress = self.progress();
        let learner = self.build_learner()?;
        let agent = self.config.algo_config.build_ppo(
            &self.algo_backend.algo,
            learner,
            &self.config.learning_hook,
            &self.config.learner_builder.optimizer,
            progress.clone(),
            self.resources.ppo_reporter.take(),
            self.resources.n_envs(),
        );
        let backend = BackendConfig::Burn(self.algo_backend.backend);
        self.finish_build(agent, backend, progress)
    }
}

impl<E: Env> OnPolicyBuilder<A2CConfig, CandleDevice, E> {
    /// Builds the configured Candle A2C training algorithm.
    ///
    /// Move the builder to the training thread before building; the algorithm is not `Send`.
    ///
    /// # Errors
    ///
    /// Returns an error if the learner, sampler, or lifecycle hooks cannot be constructed.
    ///
    /// # Panics
    ///
    /// Panics if the tensor backend cannot create enabled observation-normalization statistics.
    pub fn build(mut self) -> Result<OnPolicyAlgorithm<OnPolicyHooks<A2CCandle, E>>> {
        let progress = self.progress();
        let learner = self.build_learner()?;
        let agent = self.config.algo_config.build_a2c(
            learner,
            &self.config.learning_hook,
            &self.config.learner_builder.optimizer,
            progress.clone(),
            self.resources.a2c_reporter.take(),
            self.resources.n_envs(),
        );
        let backend = BackendConfig::Candle(self.algo_backend.backend);
        self.finish_build(agent, backend, progress)
    }
}

impl<E: Env> OnPolicyBuilder<A2CConfig, BurnBackendConfig, E> {
    /// Builds the configured Burn A2C training algorithm.
    ///
    /// Move the builder to the training thread before building; the algorithm is not `Send`.
    ///
    /// # Errors
    ///
    /// Returns an error if the learner, sampler, or lifecycle hooks cannot be constructed.
    ///
    /// # Panics
    ///
    /// Panics if the tensor backend cannot create enabled observation-normalization statistics.
    pub fn build(mut self) -> Result<OnPolicyAlgorithm<OnPolicyHooks<A2CBurn, E>>> {
        let progress = self.progress();
        let learner = self.build_learner()?;
        let agent = self.config.algo_config.build_a2c(
            learner,
            &self.config.learning_hook,
            &self.config.learner_builder.optimizer,
            progress.clone(),
            self.resources.a2c_reporter.take(),
            self.resources.n_envs(),
        );
        let backend = BackendConfig::Burn(self.algo_backend.backend);
        self.finish_build(agent, backend, progress)
    }
}

/// Default PPO builder using Candle and direct, step-bounded sampling.
pub type PPOBuilder<E, B = CandleDevice> = OnPolicyBuilder<PPOConfig, B, E>;

/// Default A2C builder using Candle and direct, step-bounded sampling.
pub type A2CBuilder<E, B = CandleDevice> = OnPolicyBuilder<A2CConfig, B, E>;

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
    pub fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, num_envs: usize) -> Result<Self> {
        Self::configured(
            env_builder,
            num_envs,
            AlgoBackendConfig {
                algo: PPOConfig::default(),
                backend: CandleDevice::Cpu,
            },
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
    pub fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, num_envs: usize) -> Result<Self> {
        Self::configured(
            env_builder,
            num_envs,
            AlgoBackendConfig {
                algo: A2CConfig,
                backend: CandleDevice::Cpu,
            },
        )
    }
}

#[cfg(feature = "gym")]
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
    pub fn gym<EB: Into<GymEnvBuilder>>(env_builder: EB, num_envs: usize) -> Result<Self> {
        Self::new(env_builder.into(), num_envs)
    }
}

#[cfg(feature = "gym")]
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
    pub fn gym<EB: Into<GymEnvBuilder>>(env_builder: EB, num_envs: usize) -> Result<Self> {
        Self::new(env_builder.into(), num_envs)
    }
}
