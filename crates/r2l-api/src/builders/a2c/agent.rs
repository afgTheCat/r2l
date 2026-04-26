use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::a2c::{A2C, A2CParams};
use r2l_core::env::ActionSpaceType;

use crate::{
    BurnBackend,
    agents::a2c::{A2CBurnAgent, A2CCandleAgent},
    builders::{
        a2c::hook::DefaultA2CHookBuilder,
        agent::{
            AgentBuilder, BurnBackend as BuilderBurnBackend, CandleBackend, OnPolicyAgentBuilder,
        },
        learning_module::{OnPolicyLearningModuleBuilder, OnPolicyLearningModuleType},
    },
    hooks::a2c::A2CStats,
};

/// Builder for A2C agents.
///
/// This is the main entry point for configuring A2C-specific agent behavior,
/// such as advantage normalization and A2C hook settings.
pub type A2CAgentBuilder = OnPolicyAgentBuilder<A2CParams, DefaultA2CHookBuilder, CandleBackend>;

/// A2C agent builder specialized to the Candle backend.
pub type A2CCandleAgentBuilder = A2CAgentBuilder;

/// A2C agent builder specialized to the Burn backend.
pub type A2CBurnAgentBuilder =
    OnPolicyAgentBuilder<A2CParams, DefaultA2CHookBuilder, BuilderBurnBackend>;

impl A2CAgentBuilder {
    /// Creates an A2C agent builder with default hyperparameters.
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultA2CHookBuilder::new(n_envs),
            params: A2CParams::default(),
            learning_module_builder: OnPolicyLearningModuleBuilder {
                policy_hidden_layers: vec![64, 64],
                value_hidden_layers: vec![64, 64],
                learning_module_type: OnPolicyLearningModuleType::Joint {
                    max_grad_norm: None,
                    params: ParamsAdamW {
                        lr: 3e-4,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-5,
                        weight_decay: 1e-4,
                    },
                },
            },
            backend: CandleBackend {
                device: Device::Cpu,
            },
        }
    }
}

impl<Backend> OnPolicyAgentBuilder<A2CParams, DefaultA2CHookBuilder, Backend> {
    /// Enables or disables advantage normalization.
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.hook_builder = self
            .hook_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    /// Sets the entropy coefficient.
    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.hook_builder = self.hook_builder.with_entropy_coeff(entropy_coeff);
        self
    }

    /// Sets the value-function loss coefficient.
    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_vf_coeff(vf_coeff);
        self
    }

    /// Sets gradient clipping for the default A2C hook.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    /// Installs a reporter channel for `A2CStats`.
    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.hook_builder = self.hook_builder.with_tx(tx);
        self
    }

    /// Sets the discount factor.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.params.gamma = gamma;
        self
    }

    /// Sets the GAE lambda parameter.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.params.lambda = lambda;
        self
    }

    /// Sets the rollout sample size used during training updates.
    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.params.sample_size = sample_size;
        self
    }
}

impl AgentBuilder for A2CAgentBuilder {
    type Agent = A2CCandleAgent;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        let device = self.backend.device.clone();
        let lm = self.learning_module_builder.build_candle(
            observation_size,
            action_size,
            action_space,
            &device,
        )?;
        let hooks = self.hook_builder.build();
        let params = self.params;
        Ok(A2CCandleAgent(A2C { lm, hooks, params }))
    }
}

impl AgentBuilder for A2CBurnAgentBuilder {
    type Agent = A2CBurnAgent<BurnBackend>;

    fn build(
        self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> anyhow::Result<Self::Agent> {
        let lm = self.learning_module_builder.build_burn::<BurnBackend>(
            observation_size,
            action_size,
            action_space,
        )?;
        let hooks = self.hook_builder.build();
        let params = self.params;
        Ok(A2CBurnAgent(A2C { lm, hooks, params }))
    }
}
