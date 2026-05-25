use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::ppo2::{PPO2, PPOParams};
use r2l_core::env::ActionSpaceType;

use crate::{
    BurnBackend,
    agents::ppo2::{PPO2BurnAgent, PPO2CandleAgent},
    builders::{
        agent2::{
            AgentBuilder2, BurnBackend2 as BuilderBurnBackend2, CandleBackend2,
            OnPolicyAgentBuilder2,
        },
        learning_module::{OnPolicyLearningModuleBuilder, OnPolicyLearningModuleType},
        ppo::hook2::DefaultPPO2HookBuilder,
    },
    hooks::ppo2::PPO2Stats,
};

/// Builder for PPO2 agents.
///
/// This is the main entry point for configuring PPO2-specific agent behavior,
/// such as clipping, advantage normalization, and PPO2 hook settings.
pub type PPO2AgentBuilder =
    OnPolicyAgentBuilder2<PPOParams, DefaultPPO2HookBuilder, CandleBackend2>;

/// PPO2 agent builder specialized to the Candle backend.
pub type PPO2CandleAgentBuilder = PPO2AgentBuilder;

/// PPO2 agent builder specialized to the Burn backend.
pub type PPO2BurnAgentBuilder =
    OnPolicyAgentBuilder2<PPOParams, DefaultPPO2HookBuilder, BuilderBurnBackend2>;

impl PPO2AgentBuilder {
    /// Creates a PPO2 agent builder with default hyperparameters.
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultPPO2HookBuilder::new(n_envs),
            params: PPOParams::default(),
            learning_module_builder: OnPolicyLearningModuleBuilder {
                policy_hidden_layers: vec![64, 64],
                value_hidden_layers: vec![64, 64],
                learning_module_type: OnPolicyLearningModuleType::Joint {
                    params: ParamsAdamW {
                        lr: 3e-4,
                        beta1: 0.9,
                        beta2: 0.999,
                        eps: 1e-5,
                        weight_decay: 1e-4,
                    },
                    max_grad_norm: None,
                },
            },
            backend: CandleBackend2 {
                device: Device::Cpu,
            },
        }
    }
}

impl<Backend> OnPolicyAgentBuilder2<PPOParams, DefaultPPO2HookBuilder, Backend> {
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

    /// Sets the target KL threshold used by the default PPO2 hook.
    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_target_kl(target_kl);
        self
    }

    /// Sets gradient clipping for the default PPO2 hook.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    /// Installs a reporter channel for `PPO2Stats`.
    pub fn with_reporter(mut self, tx: Option<Sender<PPO2Stats>>) -> Self {
        self.hook_builder = self.hook_builder.with_reporter(tx);
        self
    }

    /// Sets whether to log training progress during learning.
    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.hook_builder = self.hook_builder.with_log_progress(log_progress);
        self
    }

    /// Sets the PPO clip range.
    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.params.clip_range = clip_range;
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

impl AgentBuilder2 for PPO2AgentBuilder {
    type Agent = PPO2CandleAgent;

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
        Ok(PPO2CandleAgent(PPO2 { lm, hooks, params }))
    }
}

impl AgentBuilder2 for PPO2BurnAgentBuilder {
    type Agent = PPO2BurnAgent<BurnBackend>;

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
        Ok(PPO2BurnAgent(PPO2 { lm, hooks, params }))
    }
}
