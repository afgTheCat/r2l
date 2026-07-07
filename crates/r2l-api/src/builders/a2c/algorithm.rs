use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::a2c::A2CParams;
use r2l_core::{
    env::{Env, EnvBuilder},
    models::ActivationFunction,
    tensor::R2lTensor,
};
use r2l_gym::GymEnvBuilder;

use crate::builders::{
    a2c::{
        agent::{A2CBurnAgentBuilder, A2CCandleAgentBuilder},
        hook::DefaultA2CHookBuilder,
    },
    agent::{AgentBuilder, OnPolicyAgentBuilder},
    learning_module::OnPolicyLearningModuleType,
    on_policy::OnPolicyAlgorithmBuilder,
    sampler::{SamplerBuilder, SamplerHookBuilder, StepHookBound},
};
use crate::hooks::a2c::A2CStats;

impl<B, EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>, ST>
    OnPolicyAlgorithmBuilder<OnPolicyAgentBuilder<A2CParams, DefaultA2CHookBuilder, B>, EB, SH, ST>
where
    OnPolicyAgentBuilder<A2CParams, DefaultA2CHookBuilder, B>: AgentBuilder,
{
    /// Sets whether to log the training progress during learning.
    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.agent_builder = self.agent_builder.with_log_progress(log_progress);
        self
    }

    /// Enables or disables advantage normalization in the underlying A2C hook.
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_normalize_advantage(normalize_advantage);
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

    /// Sets optional gradient clipping in the underlying A2C hook.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    /// Installs a reporter channel for [`A2CStats`](crate::A2CStats).
    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.agent_builder = self.agent_builder.with_reporter(tx);
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

    /// Sets the hidden-layer activation function used by policy and value networks.
    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_activation_function(activation_function);
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

/// High-level A2C algorithm builder specialized to the Candle backend.
pub type A2CCandleAlgorithmBuilder<
    EB,
    SH = StepHookBound<<EB as EnvBuilder>::Env>,
    ST = crate::builders::sampler::DirectSamplerSelection,
> = OnPolicyAlgorithmBuilder<A2CCandleAgentBuilder, EB, SH, ST>;

impl A2CCandleAlgorithmBuilder<GymEnvBuilder> {
    /// Creates an A2C algorithm builder for a Gym environment.
    pub fn gym<EB: Into<GymEnvBuilder>>(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            SamplerBuilder::new(builder, n_envs),
            A2CCandleAgentBuilder::new(n_envs),
        )
    }
}

impl<EB: EnvBuilder<Env: Env<Tensor: R2lTensor>>> A2CCandleAlgorithmBuilder<EB> {
    /// Creates an A2C algorithm builder for a custom environment builder.
    pub fn new(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            SamplerBuilder::new(builder, n_envs),
            A2CCandleAgentBuilder::new(n_envs),
        )
    }
}

/// High-level A2C algorithm builder specialized to the Burn backend.
pub type A2CBurnAlgorithmBuilder<
    EB,
    SH = StepHookBound<<EB as EnvBuilder>::Env>,
    ST = crate::builders::sampler::DirectSamplerSelection,
> = OnPolicyAlgorithmBuilder<A2CBurnAgentBuilder, EB, SH, ST>;

impl<EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>, ST>
    A2CBurnAlgorithmBuilder<EB, SH, ST>
{
    /// Switches the algorithm builder to the Candle backend.
    pub fn with_candle(self, device: candle_core::Device) -> A2CCandleAlgorithmBuilder<EB, SH, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_candle(device),
            seed,
        }
    }

    /// Keeps the algorithm builder on the Burn backend.
    pub fn with_burn(self) -> A2CBurnAlgorithmBuilder<EB, SH, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_burn(),
            seed,
        }
    }
}

/// Default high-level A2C algorithm builder.
///
/// This alias uses the Candle backend by default.
pub type A2CAlgorithmBuilder<
    EB,
    SH = StepHookBound<<EB as EnvBuilder>::Env>,
    ST = crate::builders::sampler::DirectSamplerSelection,
> = A2CCandleAlgorithmBuilder<EB, SH, ST>;

impl<EB: EnvBuilder, SH: SamplerHookBuilder<Env = EB::Env>, ST>
    A2CCandleAlgorithmBuilder<EB, SH, ST>
{
    /// Switches the algorithm builder to the Candle backend.
    pub fn with_candle(self, device: Device) -> A2CCandleAlgorithmBuilder<EB, SH, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_candle(device),
            seed,
        }
    }

    /// Switches the algorithm builder to the Burn backend.
    pub fn with_burn(self) -> A2CBurnAlgorithmBuilder<EB, SH, ST> {
        let OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder,
            seed,
        } = self;
        OnPolicyAlgorithmBuilder {
            sampler_builder,
            learning_schedule,
            evaluator_builder,
            agent_builder: agent_builder.with_burn(),
            seed,
        }
    }
}
