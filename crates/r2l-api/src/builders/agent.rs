use candle_core::{Device, DeviceLocation};
use r2l_core::{
    env::Space, models::ActivationFunction, on_policy::algorithm::Agent, tensor::R2lTensor,
};
use serde::{Deserialize, Serialize, de::Error as _};

use crate::builders::learning_module::{
    AdamWParams, OnPolicyLearningModuleBuilder, OnPolicyOptimizerLayout,
};
use crate::builders::policy::PolicyBuilder;
use crate::{InferenceConfig, InferenceObservationMode};

/// Trait implemented by concrete `Agent` builders.
///
/// This trait turns high-level agent configuration into a backend-specific
/// [`Agent`](r2l_core::on_policy::algorithm::Agent) instance once the
/// environment dimensions and action-space kind are known.
pub trait AgentBuilder {
    /// Agent type produced by this builder.
    type Agent: Agent;

    /// Returns the inference configuration represented by this builder.
    ///
    /// Custom agent builders that do not support inference artifact export may
    /// retain the default `None`.
    fn inference_config(
        &self,
        _observation_mode: InferenceObservationMode,
    ) -> Option<InferenceConfig> {
        None
    }

    /// Builds the configured agent for the provided environment dimensions.
    fn build<T: R2lTensor>(
        self,
        observation_size: usize,
        action_space: Space<T>,
        seed: Option<u64>,
    ) -> anyhow::Result<Self::Agent>;
}

/// Marker type representing the Burn backend in `Agent` builders.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BurnBackendConfig;

/// Candle backend configuration used by `Agent` builders.
#[derive(Debug, Clone)]
pub struct CandleBackend {
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
    pub(crate) fn seed(&self, seed: u64) {
        if !matches!(&self.device, Device::Cpu) {
            self.device.set_seed(seed).unwrap();
        }
    }
}

/// Shared builder for on-policy `Agent` implementations.
///
/// This type provides the common configuration surface used by the concrete
/// `*2` agent builder aliases such as
/// [`PPOAgentBuilder`](crate::PPOAgentBuilder) and
/// [`A2CAgentBuilder`](crate::A2CAgentBuilder).
///
/// Most users should construct one of those aliases directly instead of naming
/// this generic type.
#[derive(Debug, Serialize, Deserialize)]
pub struct OnPolicyAgentBuilder<Params, HookBuilder, Backend> {
    pub(crate) params: Params,
    pub(crate) hook_builder: HookBuilder,
    pub(crate) learning_module_builder: OnPolicyLearningModuleBuilder,
    pub(crate) backend: Backend,
}

impl<Params, HookBuilder, Backend> OnPolicyAgentBuilder<Params, HookBuilder, Backend> {
    /// Replaces the policy architecture configuration.
    pub fn with_policy_builder(mut self, policy_builder: PolicyBuilder) -> Self {
        self.learning_module_builder.policy_builder = policy_builder;
        self
    }

    /// Switches the builder to the Candle backend.
    pub fn with_candle(
        self,
        device: Device,
    ) -> OnPolicyAgentBuilder<Params, HookBuilder, CandleBackend> {
        let OnPolicyAgentBuilder {
            params,
            hook_builder,
            learning_module_builder,
            ..
        } = self;
        OnPolicyAgentBuilder {
            params,
            hook_builder,
            learning_module_builder,
            backend: CandleBackend { device },
        }
    }

    /// Switches the builder to the Burn backend.
    pub fn with_burn(self) -> OnPolicyAgentBuilder<Params, HookBuilder, BurnBackendConfig> {
        let OnPolicyAgentBuilder {
            params,
            hook_builder,
            learning_module_builder,
            ..
        } = self;
        OnPolicyAgentBuilder {
            params,
            hook_builder,
            learning_module_builder,
            backend: BurnBackendConfig,
        }
    }

    /// Sets the hidden layer sizes used by the policy network.
    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.learning_module_builder.policy_builder.hidden_layers = policy_hidden_layers;
        self
    }

    /// Sets the hidden-layer activation function used by policy and value networks.
    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.learning_module_builder
            .policy_builder
            .activation_function = activation_function;
        self
    }

    /// Sets the initial log standard deviation for Gaussian policies.
    pub fn with_log_std_init(mut self, log_std_init: f32) -> Self {
        self.learning_module_builder.policy_builder.log_std_init = log_std_init;
        self
    }

    /// Sets the optimizer learning rate for all configured optimizers.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_module_builder.optimizer_layout = self
            .learning_module_builder
            .optimizer_layout
            .with_lr(learning_rate);
        self
    }

    /// Sets the AdamW `beta1` parameter for all configured optimizers.
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.learning_module_builder.optimizer_layout = self
            .learning_module_builder
            .optimizer_layout
            .with_beta1(beta1);
        self
    }

    /// Sets the AdamW `beta2` parameter for all configured optimizers.
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.learning_module_builder.optimizer_layout = self
            .learning_module_builder
            .optimizer_layout
            .with_beta2(beta2);
        self
    }

    /// Sets the AdamW epsilon parameter for all configured optimizers.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.learning_module_builder.optimizer_layout = self
            .learning_module_builder
            .optimizer_layout
            .with_epsilon(epsilon);
        self
    }

    /// Sets the AdamW weight decay parameter for all configured optimizers.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.learning_module_builder.optimizer_layout = self
            .learning_module_builder
            .optimizer_layout
            .with_weight_decay(weight_decay);
        self
    }

    /// Uses a joint policy-value learning module configuration.
    pub fn with_joint(mut self, max_grad_norm: Option<f32>, params: AdamWParams) -> Self {
        self.learning_module_builder.optimizer_layout = OnPolicyOptimizerLayout::Joint {
            max_grad_norm,
            params,
        };
        self
    }

    /// Uses separate optimizer settings for the policy and value modules.
    pub fn with_split(
        mut self,
        policy_max_grad_norm: Option<f32>,
        policy_params: AdamWParams,
        value_max_grad_norm: Option<f32>,
        value_params: AdamWParams,
    ) -> Self {
        self.learning_module_builder.optimizer_layout = OnPolicyOptimizerLayout::Split {
            policy_max_grad_norm,
            policy_params,
            value_max_grad_norm,
            value_params,
        };
        self
    }

    /// Sets the hidden layer sizes used by the value network.
    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.learning_module_builder.value_hidden_layers = value_hidden_layers;
        self
    }

    /// Replaces the policy/value optimizer layout.
    pub fn with_optimizer_layout(mut self, optimizer_layout: OnPolicyOptimizerLayout) -> Self {
        self.learning_module_builder.optimizer_layout = optimizer_layout;
        self
    }
}
