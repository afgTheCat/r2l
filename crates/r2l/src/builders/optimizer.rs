//! Optimizer configuration shared by the Burn and Candle builders.

use burn::{
    grad_clipping::GradientClippingConfig as BurnClipping, optim::AdamWConfig as BurnAdamW,
};
use candle_nn::ParamsAdamW;

/// Gradient clipping applied by an optimizer before each parameter update.
#[derive(Debug, Clone, Copy, Default)]
pub enum GradientClippingConfig {
    /// Leave gradients unclipped.
    #[default]
    Disabled,
    /// Clip gradients using the given maximum norm.
    /// Candle uses the norm across the optimizer's parameters; Burn clips each
    /// parameter tensor independently.
    Norm(f32),
}

impl GradientClippingConfig {
    pub(super) fn max_norm(self) -> Option<f32> {
        match self {
            Self::Disabled => None,
            Self::Norm(max_norm) => Some(max_norm),
        }
    }
}

/// Learning-rate policy applied to shared collection progress.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule {
    /// Keep the learning rate fixed throughout training.
    Constant(f64),
    /// Decay the initial learning rate to zero, including the current collection in progress.
    /// The final learning pass uses zero learning rate, including a one-rollout run.
    Linear(f64),
}

impl LearningRateSchedule {
    /// Returns the learning rate for the remaining training fraction.
    ///
    /// # Arguments
    ///
    /// * `progress_remaining` - Remaining fraction, clamped to `[0, 1]` for linear decay.
    pub(crate) fn value(self, progress_remaining: f64) -> f64 {
        match self {
            Self::Constant(learning_rate) => learning_rate,
            Self::Linear(initial_learning_rate) => {
                initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
            }
        }
    }
}

/// Complete configuration for Adam optimization with decoupled weight decay.
#[derive(Clone, Debug)]
pub struct AdamWConfig {
    /// Learning rate evaluated before each learning pass.
    pub learning_rate: LearningRateSchedule,
    /// Gradient clipping configured when the optimizer is built.
    pub gradient_clipping: GradientClippingConfig,
    /// First-moment decay coefficient.
    pub beta1: f64,
    /// Second-moment decay coefficient.
    pub beta2: f64,
    /// Numerical-stability term.
    pub eps: f64,
    /// Weight-decay coefficient.
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: LearningRateSchedule::Constant(3e-4),
            gradient_clipping: GradientClippingConfig::Disabled,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-5,
            weight_decay: 1e-4,
        }
    }
}

impl AdamWConfig {
    pub(super) fn candle_params(&self) -> ParamsAdamW {
        ParamsAdamW {
            lr: self.learning_rate.value(1.0),
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }

    pub(super) fn burn_config(&self) -> BurnAdamW {
        BurnAdamW::new()
            .with_beta_1(self.beta1 as f32)
            .with_beta_2(self.beta2 as f32)
            .with_epsilon(self.eps as f32)
            .with_weight_decay(self.weight_decay as f32)
            .with_grad_clipping(self.gradient_clipping.max_norm().map(BurnClipping::Norm))
    }
}

/// Optimizer arrangement for the policy and value networks.
#[derive(Clone, Debug)]
pub enum OptimizerConfig {
    /// One optimizer updates both networks using the same settings.
    Joint(AdamWConfig),
    /// Independent optimizers, each with its own schedule and clipping.
    Split {
        /// Configuration for the policy optimizer.
        policy: AdamWConfig,
        /// Configuration for the value optimizer.
        value: AdamWConfig,
    },
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::Joint(AdamWConfig::default())
    }
}

impl OptimizerConfig {
    pub(super) fn for_each_mut(&mut self, mut update: impl FnMut(&mut AdamWConfig)) {
        match self {
            Self::Joint(config) => update(config),
            Self::Split { policy, value } => {
                update(policy);
                update(value);
            }
        }
    }

    pub(super) fn learning_rate_schedules(&self) -> (LearningRateSchedule, LearningRateSchedule) {
        match self {
            Self::Joint(config) => (config.learning_rate, config.learning_rate),
            Self::Split { policy, value } => (policy.learning_rate, value.learning_rate),
        }
    }
}
