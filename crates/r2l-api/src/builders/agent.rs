use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_core::{env::ActionSpaceType, models::ActivationFunction, on_policy::algorithm::Agent};

use crate::builders::learning_module::{OnPolicyLearningModuleBuilder, OnPolicyLearningModuleType};

// TODO: we might not need this trait in the future
/// Trait implemented by concrete `Agent` builders.
///
/// This trait turns high-level agent configuration into a backend-specific
/// [`Agent`](r2l_core::on_policy::algorithm::Agent) instance once the
/// environment dimensions and action-space kind are known.
pub trait AgentBuilder {
    /// Agent type produced by this builder.
    type Agent: Agent;

    /// Seeds backend-specific random generators before the agent is built.
    fn seed(&self, _seed: u64) {}

    // TODO: This API is heavily in progress
    /// Builds the configured agent for the provided environment dimensions.
    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent>;
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
pub struct OnPolicyAgentBuilder<Params, HookBuilder, Backend> {
    pub(crate) params: Params,
    pub(crate) hook_builder: HookBuilder,
    pub(crate) learning_module_builder: OnPolicyLearningModuleBuilder,
    pub(crate) backend: Backend,
}

/// Marker type representing the Burn backend in `Agent` builders.
#[derive(Debug, Clone, Copy, Default)]
pub struct BurnBackend;

/// Candle backend configuration used by `Agent` builders.
#[derive(Debug, Clone)]
pub struct CandleBackend {
    pub(crate) device: Device,
}

impl CandleBackend {
    pub(crate) fn seed(&self, seed: u64) {
        if !matches!(&self.device, Device::Cpu) {
            self.device.set_seed(seed).unwrap();
        }
    }
}

impl<Params, HookBuilder, Backend> OnPolicyAgentBuilder<Params, HookBuilder, Backend> {
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
    pub fn with_burn(self) -> OnPolicyAgentBuilder<Params, HookBuilder, BurnBackend> {
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
            backend: BurnBackend,
        }
    }

    /// Sets the hidden layer sizes used by the policy network.
    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.learning_module_builder.policy_hidden_layers = policy_hidden_layers;
        self
    }

    /// Sets the hidden-layer activation function used by policy and value networks.
    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.learning_module_builder.activation_function = activation_function;
        self
    }

    /// Sets the optimizer learning rate for all configured optimizers.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_lr(learning_rate);
        self
    }

    /// Sets the AdamW `beta1` parameter for all configured optimizers.
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_beta1(beta1);
        self
    }

    /// Sets the AdamW `beta2` parameter for all configured optimizers.
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_beta2(beta2);
        self
    }

    /// Sets the AdamW epsilon parameter for all configured optimizers.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_epsilon(epsilon);
        self
    }

    /// Sets the AdamW weight decay parameter for all configured optimizers.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.learning_module_builder.learning_module_type = self
            .learning_module_builder
            .learning_module_type
            .with_weight_decay(weight_decay);
        self
    }

    /// Uses a joint policy-value learning module configuration.
    pub fn with_joint(mut self, max_grad_norm: Option<f32>, params: ParamsAdamW) -> Self {
        self.learning_module_builder.learning_module_type = OnPolicyLearningModuleType::Joint {
            max_grad_norm,
            params,
        };
        self
    }

    /// Uses separate optimizer settings for the policy and value modules.
    pub fn with_split(
        mut self,
        policy_max_grad_norm: Option<f32>,
        policy_params: ParamsAdamW,
        value_max_grad_norm: Option<f32>,
        value_params: ParamsAdamW,
    ) -> Self {
        self.learning_module_builder.learning_module_type = OnPolicyLearningModuleType::Split {
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

    /// Replaces the full learning module configuration.
    pub fn with_learning_module_type(
        mut self,
        learning_module_type: OnPolicyLearningModuleType,
    ) -> Self {
        self.learning_module_builder.learning_module_type = learning_module_type;
        self
    }
}
