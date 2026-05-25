use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::ppo2::PPOParams;
use r2l_core::env::EnvBuilder;
use r2l_core::on_policy::algorithm2::Agent2;
use r2l_gym::GymEnvBuilder;

use crate::agents::ppo2::{PPO2BurnAgent, PPO2CandleAgent};
use crate::{
    BurnBackend,
    builders::{
        agent2::{AgentBuilder2, OnPolicyAgentBuilder2},
        learning_module::OnPolicyLearningModuleType,
        on_policy2::OnPolicyAlgorithmBuilder2,
        ppo::{
            agent2::{PPO2BurnAgentBuilder, PPO2CandleAgentBuilder},
            hook2::DefaultPPO2HookBuilder,
        },
        sampler2::{Sampler2Builder, SamplerHook2Builder, StepHookBound},
    },
    hooks::ppo2::PPO2Stats,
};

impl<A: Agent2, B, EB: EnvBuilder, SH: SamplerHook2Builder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder2<
        A,
        OnPolicyAgentBuilder2<PPOParams, DefaultPPO2HookBuilder, B>,
        EB,
        SH,
    >
where
    OnPolicyAgentBuilder2<PPOParams, DefaultPPO2HookBuilder, B>: AgentBuilder2<Agent = A>,
{
    /// Enables or disables advantage normalization in the underlying PPO2 hook.
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

    /// Sets optional gradient clipping in the underlying PPO2 hook.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    /// Installs a reporter channel for [`PPO2Stats`](crate::PPO2Stats).
    pub fn with_reporter(mut self, tx: Option<Sender<PPO2Stats>>) -> Self {
        self.agent_builder = self.agent_builder.with_reporter(tx);
        self
    }

    /// Sets whether to log training progress during learning.
    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.agent_builder = self.agent_builder.with_log_progress(log_progress);
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

/// High-level PPO2 algorithm builder specialized to the Candle backend.
///
/// This builder combines environment setup, sampler construction, agent
/// construction, and default on-policy training hooks.
pub type PPO2CandleAlgorithmBuilder<EB, SH = StepHookBound<<EB as EnvBuilder>::Env>> =
    OnPolicyAlgorithmBuilder2<PPO2CandleAgent, PPO2CandleAgentBuilder, EB, SH>;

impl PPO2CandleAlgorithmBuilder<GymEnvBuilder> {
    /// Creates a PPO2 algorithm builder for a Gym environment.
    pub fn gym<EB: Into<GymEnvBuilder>>(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            Sampler2Builder::new(builder, n_envs),
            PPO2CandleAgentBuilder::new(n_envs),
        )
    }
}

impl<EB: EnvBuilder> PPO2CandleAlgorithmBuilder<EB> {
    /// Creates a PPO2 algorithm builder for a custom environment builder.
    pub fn new(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            Sampler2Builder::new(builder, n_envs),
            PPO2CandleAgentBuilder::new(n_envs),
        )
    }
}

/// High-level PPO2 algorithm builder specialized to the Burn backend.
pub type PPO2BurnAlgorithmBuilder<EB, SH = StepHookBound<<EB as EnvBuilder>::Env>> =
    OnPolicyAlgorithmBuilder2<PPO2BurnAgent<BurnBackend>, PPO2BurnAgentBuilder, EB, SH>;

impl<EB: EnvBuilder, SH: SamplerHook2Builder<Env = EB::Env>>
    PPO2BurnAlgorithmBuilder<EB, SH>
{
    /// Switches the algorithm builder to the Candle backend.
    pub fn with_candle(self, device: candle_core::Device) -> PPO2CandleAlgorithmBuilder<EB, SH> {
        let OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_candle(device),
        }
    }

    /// Keeps the algorithm builder on the Burn backend.
    pub fn with_burn(self) -> PPO2BurnAlgorithmBuilder<EB, SH> {
        let OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_burn(),
        }
    }
}

/// Default high-level PPO2 algorithm builder.
///
/// This alias uses the Candle backend by default.
pub type PPO2AlgorithmBuilder<EB, SH = StepHookBound<<EB as EnvBuilder>::Env>> =
    PPO2CandleAlgorithmBuilder<EB, SH>;

impl<EB: EnvBuilder, SH: SamplerHook2Builder<Env = EB::Env>>
    PPO2CandleAlgorithmBuilder<EB, SH>
{
    /// Switches the algorithm builder to the Candle backend.
    pub fn with_candle(self, device: Device) -> PPO2CandleAlgorithmBuilder<EB, SH> {
        let OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_candle(device),
        }
    }

    /// Switches the algorithm builder to the Burn backend.
    pub fn with_burn(self) -> PPO2BurnAlgorithmBuilder<EB, SH> {
        let OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
        } = self;
        OnPolicyAlgorithmBuilder2 {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_burn(),
        }
    }
}
