use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::ppo::{PPO, PPOParams};
use r2l_core::env::ActionSpaceType;

use crate::{
    BurnBackend,
    agents::ppo::{PPOBurnAgent, PPOCandleAgent},
    builders::{
        agent::{
            AgentBuilder, BurnBackend as BuilderBurnBackend, CandleBackend, OnPolicyAgentBuilder,
        },
        learning_module::{OnPolicyLearningModuleBuilder, OnPolicyLearningModuleType},
        ppo::hook::DefaultPPOHookBuilder,
    },
    hooks::ppo::PPOStats,
};

pub type PPOAgentBuilder = OnPolicyAgentBuilder<PPOParams, DefaultPPOHookBuilder, CandleBackend>;
pub type PPOCandleAgentBuilder = PPOAgentBuilder;
pub type PPOBurnAgentBuilder =
    OnPolicyAgentBuilder<PPOParams, DefaultPPOHookBuilder, BuilderBurnBackend>;

impl PPOAgentBuilder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            hook_builder: DefaultPPOHookBuilder::new(n_envs),
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
            backend: CandleBackend {
                device: Device::Cpu,
            },
        }
    }
}

impl<Backend> OnPolicyAgentBuilder<PPOParams, DefaultPPOHookBuilder, Backend> {
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.hook_builder = self
            .hook_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.hook_builder = self.hook_builder.with_entropy_coeff(entropy_coeff);
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_vf_coeff(vf_coeff);
        self
    }

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_target_kl(target_kl);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.hook_builder = self.hook_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.hook_builder = self.hook_builder.with_tx(tx);
        self
    }

    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.params.clip_range = clip_range;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.params.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.params.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.params.sample_size = sample_size;
        self
    }
}

impl AgentBuilder for PPOAgentBuilder {
    type Agent = PPOCandleAgent;

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
        Ok(PPOCandleAgent(PPO { lm, hooks, params }))
    }
}

impl AgentBuilder for PPOBurnAgentBuilder {
    type Agent = PPOBurnAgent<BurnBackend>;

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
        Ok(PPOBurnAgent(PPO { lm, hooks, params }))
    }
}
