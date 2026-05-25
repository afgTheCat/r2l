use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::a2c2::{A2C2, A2CParams};
use r2l_core::env::ActionSpaceType;

use crate::{
    BurnBackend,
    agents::a2c2::{A2C2BurnAgent, A2C2CandleAgent},
    builders::{
        a2c::hook2::DefaultA2CHook2Builder,
        agent2::{
            AgentBuilder2, BurnBackend2 as BuilderBurnBackend2, CandleBackend2,
            OnPolicyAgentBuilder2,
        },
        learning_module::{OnPolicyLearningModuleBuilder, OnPolicyLearningModuleType},
    },
    hooks::a2c2::A2CStats,
};

/// Builder for A2C2 agents.
///
/// This is the main entry point for configuring A2C2-specific agent behavior,
/// such as advantage normalization and A2C2 hook settings.
pub type A2C2AgentBuilder =
    OnPolicyAgentBuilder2<A2CParams, DefaultA2CHook2Builder, CandleBackend2>;

/// A2C2 agent builder specialized to the Candle backend.
pub type A2C2CandleAgentBuilder = A2C2AgentBuilder;

/// A2C2 agent builder specialized to the Burn backend.
pub type A2C2BurnAgentBuilder =
    OnPolicyAgentBuilder2<A2CParams, DefaultA2CHook2Builder, BuilderBurnBackend2>;

impl A2C2AgentBuilder {
    /// Creates an A2C2 agent builder with default hyperparameters.
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultA2CHook2Builder::new(n_envs),
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
            backend: CandleBackend2 {
                device: Device::Cpu,
            },
        }
    }
}

impl<Backend> OnPolicyAgentBuilder2<A2CParams, DefaultA2CHook2Builder, Backend> {
    /// Sets whether to log the training progress during learning.
    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.hook_builder = self.hook_builder.with_log_progress(log_progress);
        self
    }

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

    /// Sets gradient clipping for the default A2C2 hook.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    /// Installs a reporter channel for `A2CStats`.
    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.hook_builder = self.hook_builder.with_reporter(tx);
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

impl AgentBuilder2 for A2C2AgentBuilder {
    type Agent = A2C2CandleAgent;

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
        Ok(A2C2CandleAgent(A2C2 { lm, hooks, params }))
    }
}

impl AgentBuilder2 for A2C2BurnAgentBuilder {
    type Agent = A2C2BurnAgent<BurnBackend>;

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
        Ok(A2C2BurnAgent(A2C2 { lm, hooks, params }))
    }
}
