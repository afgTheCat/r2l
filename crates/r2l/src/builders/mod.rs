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

use candle_core::Device;
use networks::NetworkConfig;
pub use on_policy_hook::TrainingArtifactsConfig;
pub use optimizer::{AdamWConfig, GradientClippingConfig, LearningRateSchedule, OptimizerConfig};
use r2l_agents::on_policy_algorithms::{
    a2c::{A2C, A2CHook},
    ppo::{PPO, PPOHook},
};
use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    error::Error,
    models::{ActivationFunction, ToSafetensors},
    on_policy::{
        algorithm::{Agent, OnPolicyAlgorithm, OnPolicyRuntime, Sampler},
        learning_module::OnPolicyLearner,
    },
};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearner as BurnPolicyValueLearner;
use r2l_distributions::learning_modules::candle_lm::PolicyValueLearner as CandlePolicyValueLearner;
#[cfg(feature = "gym")]
use r2l_gym::{GymEnv, GymEnvBuilder};
use r2l_sampler::{DirectSampler, SamplerExecutionMode, StagedSampler};
pub use sampler::ObsNormalizerConfig;

use crate::{
    A2CRolloutStats, EpisodeBoundHook, OnPolicyControlHandle, PPORolloutStats, StepBoundHook,
    TrainingLimit,
    backend::{Backend, BurnBackendConfig, CandleBackend},
    builders::{
        algorithm::AlgoConfig,
        learner::{LearnerConfig, LearningHookConfig},
        on_policy_hook::OnPolicyHookConfig,
        sampler::{SamplerConfiguration, SamplerSetup},
    },
    hooks::{
        learning::{A2CLearningHook, ClipRangeSchedule, PPOLearningHook},
        on_policy::{OnPolicyTrainingHooks, commands::on_policy_control_channel},
        progress::{SharedTrainingProgress, TrainingProgress},
    },
};

/// PPO agent produced by a Candle-backed algorithm builder.
pub type PPOCandle = PPO<CandlePolicyValueLearner, PPOLearningHook<CandlePolicyValueLearner>>;
/// PPO agent produced by a Burn-backed algorithm builder.
pub type PPOBurn = PPO<BurnPolicyValueLearner, PPOLearningHook<BurnPolicyValueLearner>>;
/// A2C agent produced by a Candle-backed algorithm builder.
pub type A2CCandle = A2C<CandlePolicyValueLearner, A2CLearningHook<CandlePolicyValueLearner>>;
/// A2C agent produced by a Burn-backed algorithm builder.
pub type A2CBurn = A2C<BurnPolicyValueLearner, A2CLearningHook<BurnPolicyValueLearner>>;

struct Builder<E: Env + 'static> {
    backend_configuration: Backend,
    seed: Option<u64>,

    // for the hooks
    training_limit: TrainingLimit,
    hook_config: OnPolicyHookConfig,

    // for the agent
    learner_builder: LearnerConfig,
    algo_config: AlgoConfig,
    learning_hook: LearningHookConfig,
    ppo_reporter: Option<Sender<PPORolloutStats>>,
    a2c_reporter: Option<Sender<A2CRolloutStats>>,

    // for the sampler
    sampler_configuration: SamplerConfiguration<E>,
}

impl<E: Env> Builder<E> {
    fn new<EB: EnvBuilder<Env = E>>(
        env_builder: EB,
        n_envs: usize,
        algo_config: AlgoConfig,
        backend_configuration: Backend,
        sampler_setup: SamplerSetup<E>,
        normalize_advantage: bool,
    ) -> Result<Self, Error> {
        let env_builder = EnvBuilderType::homogeneous(env_builder, n_envs)?;
        let env_description = env_builder.env_description()?;
        let learner_builder = LearnerConfig::new(&env_description)?;
        Ok(Self {
            sampler_configuration: SamplerConfiguration {
                setup: sampler_setup,
                execution_mode: SamplerExecutionMode::MultiThreaded,
                env_builder,
            },
            backend_configuration,
            algo_config,
            training_limit: TrainingLimit::rollouts(50),
            hook_config: OnPolicyHookConfig::default(),
            learner_builder,
            learning_hook: LearningHookConfig::new(normalize_advantage),
            ppo_reporter: None,
            a2c_reporter: None,
            seed: None,
        })
    }

    fn build_candle_learner(&self) -> Result<CandlePolicyValueLearner, Error> {
        let Backend::Candle(backend) = &self.backend_configuration else {
            unreachable!("Candle agent type must use Candle backend configuration")
        };
        self.learner_builder.build_candle_learner(&backend.device)
    }

    fn build_burn_learner(&self) -> Result<BurnPolicyValueLearner, Error> {
        let Backend::Burn(_) = self.backend_configuration else {
            unreachable!("Burn agent type must use Burn backend configuration")
        };
        self.learner_builder.build_burn_learner()
    }

    fn progress(&self) -> SharedTrainingProgress {
        TrainingProgress::shared(
            self.training_limit,
            self.sampler_configuration.rollout_mode(),
            self.sampler_configuration.n_envs(),
        )
    }

    fn ppo_candle_agent(&mut self, progress: SharedTrainingProgress) -> Result<PPOCandle, Error> {
        let learner = self.build_candle_learner()?;
        Ok(self.algo_config.build_ppo(
            learner,
            &self.learning_hook,
            &self.learner_builder.optimizer,
            progress,
            self.ppo_reporter.take(),
            self.sampler_configuration.n_envs(),
        ))
    }

    fn ppo_burn_agent(&mut self, progress: SharedTrainingProgress) -> Result<PPOBurn, Error> {
        let learner = self.build_burn_learner()?;
        Ok(self.algo_config.build_ppo(
            learner,
            &self.learning_hook,
            &self.learner_builder.optimizer,
            progress,
            self.ppo_reporter.take(),
            self.sampler_configuration.n_envs(),
        ))
    }

    fn a2c_candle_agent(&mut self, progress: SharedTrainingProgress) -> Result<A2CCandle, Error> {
        let learner = self.build_candle_learner()?;
        Ok(self.algo_config.build_a2c(
            learner,
            &self.learning_hook,
            &self.learner_builder.optimizer,
            progress,
            self.a2c_reporter.take(),
            self.sampler_configuration.n_envs(),
        ))
    }

    fn a2c_burn_agent(&mut self, progress: SharedTrainingProgress) -> Result<A2CBurn, Error> {
        let learner = self.build_burn_learner()?;
        Ok(self.algo_config.build_a2c(
            learner,
            &self.learning_hook,
            &self.learner_builder.optimizer,
            progress,
            self.a2c_reporter.take(),
            self.sampler_configuration.n_envs(),
        ))
    }
}

struct Config<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor> + 'static> {
    build_agent: fn(&mut Builder<E>, SharedTrainingProgress) -> Result<A, Error>,
    build_sampler: fn(&SamplerConfiguration<E>, SharedTrainingProgress) -> Result<S, Error>,
}

/// Configures and builds a complete PPO or A2C training algorithm.
///
/// Start with [`PPOBuilder`] or [`A2CBuilder`]. Some methods,
/// such as backend and sampler selection, return a builder with different
/// generic arguments. The concrete state components are publicly named so such
/// builders can be stored in and passed to user functions.
///
/// Both entry points default to Candle on the CPU, multi-threaded direct
/// sampling with 1,024 steps per environment, and a training limit of 50
/// rollouts. Shared learning defaults include `gamma = 0.98`, `lambda = 0.8`,
/// minibatches of 64 samples, and a joint Adam optimizer with decoupled weight
/// decay and a learning rate of `3e-4`.
#[must_use]
pub struct OnPolicyBuilder<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor> + 'static> {
    builder: Builder<E>,
    config: Config<A, S, E>,
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>>
    OnPolicyBuilder<A, S, E>
{
    fn configured<EB: EnvBuilder<Env = E>>(
        env_builder: EB,
        n_envs: usize,
        algo_config: AlgoConfig,
        backend_configuration: Backend,
        sampler_setup: SamplerSetup<E>,
        normalize_advantage: bool,
        build_agent: fn(&mut Builder<E>, SharedTrainingProgress) -> Result<A, Error>,
        build_sampler: fn(&SamplerConfiguration<E>, SharedTrainingProgress) -> Result<S, Error>,
    ) -> Result<Self, Error> {
        Ok(Self {
            builder: Builder::new(
                env_builder,
                n_envs,
                algo_config,
                backend_configuration,
                sampler_setup,
                normalize_advantage,
            )?,
            config: Config {
                build_agent,
                build_sampler,
            },
        })
    }

    fn with_agent<A2: Agent>(
        self,
        build_agent: fn(&mut Builder<E>, SharedTrainingProgress) -> Result<A2, Error>,
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
        build_sampler: fn(&SamplerConfiguration<E>, SharedTrainingProgress) -> Result<S2, Error>,
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
        self.builder.hook_config.training_artifacts = Some(config);
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
        self.builder.hook_config.avg_reward_threshold = avg_reward_threshold;
        self
    }

    /// Enables external control of the configured training algorithm.
    pub fn with_control(mut self) -> (Self, OnPolicyControlHandle) {
        let (control_endpoint, control_handle) = on_policy_control_channel();
        self.builder.hook_config.control_endpoint = Some(control_endpoint);
        (self, control_handle)
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
        self.builder
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
        self.builder.seed = Some(seed);
        self
    }

    /// Selects single-threaded or multi-threaded environment execution.
    ///
    /// # Arguments
    ///
    /// * `execution_mode` - How sampler environment workers should be executed.
    pub fn with_execution_mode(mut self, execution_mode: SamplerExecutionMode) -> Self {
        self.builder.sampler_configuration.execution_mode = execution_mode;
        self
    }

    /// Selects the policy architecture. CNN construction currently requires Burn.
    /// CNNs reshape flat observations using the environment's `[channels, height, width]` shape.
    ///
    /// # Arguments
    ///
    /// * `network` - Hidden architecture; observation shape and action-space outputs are inferred.
    pub fn with_policy_network(mut self, network: NetworkConfig) -> Self {
        self.builder.learner_builder.policy_config.network = network;
        self
    }

    /// Selects the value architecture independently of the policy. CNNs currently require Burn.
    /// CNNs reshape flat observations using the environment's `[channels, height, width]` shape.
    ///
    /// # Arguments
    ///
    /// * `network` - Hidden architecture; observation shape is inferred and output width is one.
    pub fn with_value_network(mut self, network: NetworkConfig) -> Self {
        self.builder.learner_builder.value_network = network;
        self
    }

    /// Sets the hidden-layer widths of the policy network.
    ///
    /// # Arguments
    ///
    /// * `policy_hidden_layers` - Dense hidden-layer widths, including the dense layers after a CNN.
    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.builder
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
        self.builder
            .learner_builder
            .policy_config
            .network
            .mlp_config_mut()
            .activation = activation_function;
        self.builder
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
        self.builder.learner_builder.policy_config.log_std_init = log_std_init;
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
        self.builder.learner_builder.policy_config.sde = Some(config);
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
        self.builder
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
        self.builder
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
        self.builder
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
        self.builder
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
        self.builder.learner_builder.optimizer = config;
        self
    }

    /// Sets the hidden-layer widths of the value network.
    ///
    /// # Arguments
    ///
    /// * `value_hidden_layers` - Dense hidden-layer widths, including the dense layers after a CNN.
    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.builder
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
        self.builder.learning_hook.normalize_advantage = normalize_advantage;
        self
    }

    /// Sets the entropy term coefficient in the training loss.
    ///
    /// # Arguments
    ///
    /// * `entropy_coeff` - Multiplier applied to the entropy term.
    pub fn with_entropy_coefficient(mut self, entropy_coeff: f32) -> Self {
        self.builder.learning_hook.entropy_coeff = entropy_coeff;
        self
    }

    /// Sets the value-function loss coefficient, which defaults to `1.0`.
    ///
    /// # Arguments
    ///
    /// * `vf_coeff` - Value-loss multiplier; `1.0` uses the unscaled loss.
    pub fn with_value_loss_coefficient(mut self, vf_coeff: f32) -> Self {
        self.builder.learning_hook.vf_coeff = vf_coeff;
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
        self.builder
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
        match &mut self.builder.algo_config {
            AlgoConfig::PPO {
                log_progress: configured,
                ..
            }
            | AlgoConfig::A2C {
                log_progress: configured,
                ..
            } => *configured = log_progress,
        }
        self
    }

    /// Sets the reward discount factor.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to compute returns.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        match &mut self.builder.algo_config {
            AlgoConfig::PPO {
                gamma: configured, ..
            }
            | AlgoConfig::A2C {
                gamma: configured, ..
            } => *configured = gamma,
        }
        self
    }

    /// Sets the generalized advantage-estimation lambda.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Bias-variance tradeoff used by generalized advantage estimation.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        match &mut self.builder.algo_config {
            AlgoConfig::PPO {
                lambda: configured, ..
            }
            | AlgoConfig::A2C {
                lambda: configured, ..
            } => *configured = lambda,
        }
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
        match &mut self.builder.algo_config {
            AlgoConfig::PPO {
                sample_size: configured,
                ..
            }
            | AlgoConfig::A2C {
                sample_size: configured,
                ..
            } => *configured = sample_size,
        }
        self
    }

    /// Builds the configured agent, sampler, and training lifecycle hooks.
    ///
    /// The algorithm shares progress on its owning thread and is not `Send`.
    /// For background training, move the builder to that thread before calling `build`.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured training algorithm cannot be constructed.
    #[allow(clippy::type_complexity)]
    pub fn build(
        mut self,
    ) -> Result<OnPolicyAlgorithm<A, S, OnPolicyTrainingHooks<A, S, E>>, Error> {
        let progress = self.builder.progress();
        if let Some(seed) = self.builder.seed {
            self.builder.backend_configuration.seed(seed)?;
        }
        let agent = (self.config.build_agent)(&mut self.builder, progress.clone())?;
        let sampler =
            (self.config.build_sampler)(&self.builder.sampler_configuration, progress.clone())?;
        let hooks = self.builder.hook_config.build(
            &self.builder.sampler_configuration,
            &self.builder.learner_builder.policy_config,
            &self.builder.backend_configuration,
            progress,
        )?;
        let runtime = OnPolicyRuntime::new(agent, sampler);
        Ok(OnPolicyAlgorithm::new(runtime, hooks))
    }
}

impl<S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyBuilder<PPOCandle, S, E> {
    /// Uses Candle on `device` for PPO learning.
    ///
    /// # Arguments
    ///
    /// * `device` - Candle device on which policy and value learning will run.
    pub fn with_candle(mut self, device: Device) -> Self {
        self.builder.backend_configuration = Backend::Candle(CandleBackend { device });
        self
    }

    /// Switches PPO learning to the default Burn backend.
    pub fn with_burn(mut self) -> OnPolicyBuilder<PPOBurn, S, E> {
        self.builder.backend_configuration = Backend::Burn(BurnBackendConfig);
        self.with_agent(Builder::ppo_burn_agent)
    }
}

impl<S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyBuilder<PPOBurn, S, E> {
    /// Switches PPO learning to Candle on `device`.
    ///
    /// # Arguments
    ///
    /// * `device` - Candle device on which policy and value learning will run.
    pub fn with_candle(mut self, device: Device) -> OnPolicyBuilder<PPOCandle, S, E> {
        self.builder.backend_configuration = Backend::Candle(CandleBackend { device });
        self.with_agent(Builder::ppo_candle_agent)
    }

    /// Keeps PPO learning on the default Burn backend.
    pub fn with_burn(mut self) -> Self {
        self.builder.backend_configuration = Backend::Burn(BurnBackendConfig);
        self
    }
}

impl<M, S, E> OnPolicyBuilder<PPO<M, PPOLearningHook<M>>, S, E>
where
    M: OnPolicyLearner,
    M::Policy: ToSafetensors,
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
        self.builder.ppo_reporter = tx;
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
        let AlgoConfig::PPO {
            total_epochs: configured,
            ..
        } = &mut self.builder.algo_config
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        *configured =
            NonZeroUsize::new(total_epochs).expect("total epochs must be greater than zero");
        self
    }

    /// Sets the optional KL-divergence threshold for stopping PPO epochs early.
    ///
    /// # Arguments
    ///
    /// * `target_kl` - KL-divergence threshold, or `None` to disable early stopping.
    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        let AlgoConfig::PPO {
            target_kl: configured,
            ..
        } = &mut self.builder.algo_config
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
    pub fn with_clip_range(self, clip_range: f32) -> Self {
        self.with_clip_range_schedule(ClipRangeSchedule::Constant(clip_range))
    }

    /// Sets the PPO policy-ratio clipping schedule.
    ///
    /// # Arguments
    ///
    /// * `clip_range_schedule` - Schedule applied to the clipping range as training progresses.
    pub fn with_clip_range_schedule(mut self, clip_range_schedule: ClipRangeSchedule) -> Self {
        let AlgoConfig::PPO {
            clip_range_schedule: configured,
            ..
        } = &mut self.builder.algo_config
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        *configured = clip_range_schedule;
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
        self.builder.backend_configuration = Backend::Candle(CandleBackend { device });
        self
    }

    /// Switches A2C learning to the default Burn backend.
    pub fn with_burn(mut self) -> OnPolicyBuilder<A2CBurn, S, E> {
        self.builder.backend_configuration = Backend::Burn(BurnBackendConfig);
        self.with_agent(Builder::a2c_burn_agent)
    }
}

impl<S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyBuilder<A2CBurn, S, E> {
    /// Switches A2C learning to Candle on `device`.
    ///
    /// # Arguments
    ///
    /// * `device` - Candle device on which policy and value learning will run.
    pub fn with_candle(mut self, device: Device) -> OnPolicyBuilder<A2CCandle, S, E> {
        self.builder.backend_configuration = Backend::Candle(CandleBackend { device });
        self.with_agent(Builder::a2c_candle_agent)
    }

    /// Keeps A2C learning on the default Burn backend.
    pub fn with_burn(mut self) -> Self {
        self.builder.backend_configuration = Backend::Burn(BurnBackendConfig);
        self
    }
}

impl<M, S, E> OnPolicyBuilder<A2C<M, A2CLearningHook<M>>, S, E>
where
    M: OnPolicyLearner,
    M::Policy: ToSafetensors,
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
        self.builder.a2c_reporter = tx;
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
    ///
    /// # Panics
    ///
    /// Panics if `rollout_steps` is zero.
    pub fn with_rollout_steps(mut self, rollout_steps: usize) -> Self {
        let SamplerSetup::DirectStep {
            rollout_steps: configured,
            ..
        } = &mut self.builder.sampler_configuration.setup
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        *configured =
            NonZeroUsize::new(rollout_steps).expect("rollout steps must be greater than zero");
        self
    }

    /// Normalizes discounted rewards and clips them to `clip_reward`.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to track discounted returns.
    /// * `clip_reward` - Absolute limit applied to normalized rewards.
    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        self.builder
            .sampler_configuration
            .set_reward_normalizer(gamma, clip_reward);
        self
    }

    /// Selects staged sampling and configures observation normalization.
    ///
    /// Enabled normalization can optionally clip the normalized values. Disabled
    /// normalization retains staged sampling without applying a normalizer.
    ///
    /// # Arguments
    ///
    /// * `config` - Whether to normalize observations and the optional absolute clipping limit.
    ///
    /// # Panics
    ///
    /// Panics if the tensor backend cannot create the normalization statistics.
    #[allow(clippy::type_complexity)]
    pub fn with_observation_normalizer(
        mut self,
        config: ObsNormalizerConfig,
    ) -> OnPolicyBuilder<A, StagedSampler<E, StepBoundHook<E>>, E> {
        self.builder.sampler_configuration = self
            .builder
            .sampler_configuration
            .with_observation_normalizer(config);
        self.with_sampler(SamplerConfiguration::staged_sampler_step_bound)
    }

    /// Selects direct sampling bounded by completed episodes per environment.
    ///
    /// # Arguments
    ///
    /// * `rollout_episodes` - Number of completed episodes collected from each environment per
    ///   rollout.
    ///
    /// # Panics
    ///
    /// Panics if `rollout_episodes` is zero.
    pub fn with_rollout_episodes(
        mut self,
        rollout_episodes: usize,
    ) -> OnPolicyBuilder<A, DirectSampler<E, EpisodeBoundHook<E>>, E> {
        let SamplerSetup::DirectStep { .. } = self.builder.sampler_configuration.setup else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        let rollout_episodes = NonZeroUsize::new(rollout_episodes)
            .expect("rollout episodes must be greater than zero");
        self.builder.sampler_configuration.setup = SamplerSetup::DirectEpisode { rollout_episodes };
        self.with_sampler(SamplerConfiguration::direct_sampler_episode_bound)
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
    ///
    /// # Panics
    ///
    /// Panics if `rollout_steps` is zero.
    pub fn with_rollout_steps(mut self, rollout_steps: usize) -> Self {
        let SamplerSetup::StagedStep {
            rollout_steps: configured,
            ..
        } = &mut self.builder.sampler_configuration.setup
        else {
            unreachable!("staged step-bound sampler type must use matching configuration")
        };
        *configured =
            NonZeroUsize::new(rollout_steps).expect("rollout steps must be greater than zero");
        self
    }

    /// Normalizes discounted rewards and clips them to `clip_reward`.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Discount factor used to track discounted returns.
    /// * `clip_reward` - Absolute limit applied to normalized rewards.
    pub fn with_reward_normalizer(mut self, gamma: f32, clip_reward: f32) -> Self {
        self.builder
            .sampler_configuration
            .set_reward_normalizer(gamma, clip_reward);
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
            AlgoConfig::PPO {
                gamma: 0.98,
                lambda: 0.8,
                sample_size: NonZeroUsize::new(64).unwrap(),
                total_epochs: NonZeroUsize::new(10).unwrap(),
                target_kl: None,
                clip_range_schedule: ClipRangeSchedule::Constant(0.2),
                log_progress: true,
            },
            Backend::Candle(CandleBackend {
                device: Device::Cpu,
            }),
            SamplerSetup::DirectStep {
                rollout_steps: NonZeroUsize::new(1024).unwrap(),
                reward_normalizer: None,
            },
            true,
            Builder::ppo_candle_agent,
            SamplerConfiguration::direct_sampler_step_bound,
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
            AlgoConfig::A2C {
                gamma: 0.98,
                lambda: 0.8,
                sample_size: NonZeroUsize::new(64).unwrap(),
                log_progress: true,
            },
            Backend::Candle(CandleBackend {
                device: Device::Cpu,
            }),
            SamplerSetup::DirectStep {
                rollout_steps: NonZeroUsize::new(1024).unwrap(),
                reward_normalizer: None,
            },
            true,
            Builder::a2c_candle_agent,
            SamplerConfiguration::direct_sampler_step_bound,
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
    pub fn gym<EB: Into<GymEnvBuilder>>(env_builder: EB, num_envs: usize) -> Result<Self, Error> {
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
    pub fn gym<EB: Into<GymEnvBuilder>>(env_builder: EB, num_envs: usize) -> Result<Self, Error> {
        Self::new(env_builder.into(), num_envs)
    }
}
