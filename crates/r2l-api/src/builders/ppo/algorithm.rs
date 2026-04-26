use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::ppo::PPOParams;
use r2l_core::env::{EnvBuilder, TensorOfEnvBuilder};
use r2l_core::on_policy::algorithm::Agent;
use r2l_gym::GymEnvBuilder;
use r2l_sampler::{StepTrajectoryBound, TrajectoryBound};

use crate::agents::ppo::{PPOBurnAgent, PPOCandleAgent};
use crate::{
    BurnBackend,
    builders::{
        agent::{AgentBuilder, OnPolicyAgentBuilder},
        learning_module::OnPolicyLearningModuleType,
        on_policy::OnPolicyAlgorithmBuilder,
        ppo::{
            agent::{PPOBurnAgentBuilder, PPOCandleAgentBuilder},
            hook::DefaultPPOHookBuilder,
        },
        sampler::SamplerBuilder,
    },
    hooks::ppo::PPOStats,
};

impl<A, M, EB, BD> OnPolicyAlgorithmBuilder<A, OnPolicyAgentBuilder<PPOParams, DefaultPPOHookBuilder, M>, EB, BD>
where
    A: Agent,
    EB: EnvBuilder,
    BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>>,
    OnPolicyAgentBuilder<PPOParams, DefaultPPOHookBuilder, M>: AgentBuilder<Agent = A>,
{
    /// Enables or disables advantage normalization in the underlying PPO hook.
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    /// Sets the maximum number of PPO epochs per rollout.
    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_total_epochs(total_epochs);
        self
    }

    /// Sets the entropy coefficient.
    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.agent_builder = self.agent_builder.with_entropy_coeff(entropy_coeff);
        self
    }

    /// Sets the optional value-function loss coefficient.
    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_vf_coeff(vf_coeff);
        self
    }

    /// Sets the optional target KL threshold used for early stopping.
    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_target_kl(target_kl);
        self
    }

    /// Sets optional gradient clipping in the underlying PPO hook.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    /// Installs a reporter channel for [`PPOStats`](crate::PPOStats).
    pub fn with_reporter(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.agent_builder = self.agent_builder.with_reporter(tx);
        self
    }

    /// Sets the PPO clip range.
    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.agent_builder = self.agent_builder.with_clip_range(clip_range);
        self
    }

    /// Sets the discount factor.
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.agent_builder = self.agent_builder.with_gamma(gamma);
        self
    }

    /// Sets the GAE lambda parameter.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.agent_builder = self.agent_builder.with_lambda(lambda);
        self
    }

    /// Sets the rollout sample size used during training updates.
    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.agent_builder = self.agent_builder.with_sample_size(sample_size);
        self
    }

    /// Sets the hidden layer sizes used by the policy network.
    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_policy_hidden_layers(policy_hidden_layers);
        self
    }

    /// Sets the optimizer learning rate for all configured optimizers.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.agent_builder = self.agent_builder.with_learning_rate(learning_rate);
        self
    }

    /// Sets the AdamW `beta1` parameter for all configured optimizers.
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.agent_builder = self.agent_builder.with_beta1(beta1);
        self
    }

    /// Sets the AdamW `beta2` parameter for all configured optimizers.
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.agent_builder = self.agent_builder.with_beta2(beta2);
        self
    }

    /// Sets the AdamW epsilon parameter for all configured optimizers.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.agent_builder = self.agent_builder.with_epsilon(epsilon);
        self
    }

    /// Sets the AdamW weight decay parameter for all configured optimizers.
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.agent_builder = self.agent_builder.with_weight_decay(weight_decay);
        self
    }

    /// Uses a joint policy-value learning module configuration.
    pub fn with_joint(mut self, max_grad_norm: Option<f32>, params: ParamsAdamW) -> Self {
        self.agent_builder = self.agent_builder.with_joint(max_grad_norm, params);
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
        self.agent_builder = self.agent_builder.with_split(
            policy_max_grad_norm,
            policy_params,
            value_max_grad_norm,
            value_params,
        );
        self
    }

    /// Sets the hidden layer sizes used by the value network.
    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_value_hidden_layers(value_hidden_layers);
        self
    }

    /// Replaces the full learning module configuration.
    pub fn with_learning_module_type(
        mut self,
        learning_module_type: OnPolicyLearningModuleType,
    ) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_learning_module_type(learning_module_type);
        self
    }
}

/// High-level PPO algorithm builder specialized to the Candle backend.
///
/// This builder combines environment setup, sampler construction, agent
/// construction, and default on-policy training hooks.
pub type PPOCandleAlgorithmBuilder<EB, BD = StepTrajectoryBound<TensorOfEnvBuilder<EB>>> =
    OnPolicyAlgorithmBuilder<PPOCandleAgent, PPOCandleAgentBuilder, EB, BD>;

impl PPOCandleAlgorithmBuilder<GymEnvBuilder> {
    /// Creates a PPO algorithm builder for a Gym environment.
    pub fn gym<EB: Into<GymEnvBuilder>>(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            SamplerBuilder::new(builder, n_envs),
            PPOCandleAgentBuilder::new(n_envs),
        )
    }
}

impl<EB: EnvBuilder> PPOCandleAlgorithmBuilder<EB> {
    /// Creates a PPO algorithm builder for a custom environment builder.
    pub fn new(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            SamplerBuilder::new(builder, n_envs),
            PPOCandleAgentBuilder::new(n_envs),
        )
    }
}

/// High-level PPO algorithm builder specialized to the Burn backend.
pub type PPOBurnAlgorithmBuilder<EB, BD = StepTrajectoryBound<TensorOfEnvBuilder<EB>>> =
    OnPolicyAlgorithmBuilder<PPOBurnAgent<BurnBackend>, PPOBurnAgentBuilder, EB, BD>;

impl<EB: EnvBuilder> PPOBurnAlgorithmBuilder<EB> {
    /// Switches the algorithm builder to the Candle backend.
    pub fn with_candle(self, device: candle_core::Device) -> PPOCandleAlgorithmBuilder<EB> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_candle(device),
        }
    }

    /// Keeps the algorithm builder on the Burn backend.
    pub fn with_burn(self) -> PPOBurnAlgorithmBuilder<EB> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_burn(),
        }
    }
}

/// Default high-level PPO algorithm builder.
///
/// This alias uses the Candle backend by default.
pub type PPOAlgorithmBuilder<EB, BD = StepTrajectoryBound<TensorOfEnvBuilder<EB>>> =
    PPOCandleAlgorithmBuilder<EB, BD>;

impl<EB: EnvBuilder> PPOCandleAlgorithmBuilder<EB> {
    /// Switches the algorithm builder to the Candle backend.
    pub fn with_candle(self, device: Device) -> PPOCandleAlgorithmBuilder<EB> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_candle(device),
        }
    }

    /// Switches the algorithm builder to the Burn backend.
    pub fn with_burn(self) -> PPOBurnAlgorithmBuilder<EB> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_burn(),
        }
    }
}
