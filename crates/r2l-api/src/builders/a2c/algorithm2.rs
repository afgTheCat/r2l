use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_agents::on_policy_algorithms::a2c2::A2CParams;
use r2l_core::env::EnvBuilder;
use r2l_core::on_policy::algorithm2::Agent2;
use r2l_gym::GymEnvBuilder;

use crate::agents::a2c2::{A2C2BurnAgent, A2C2CandleAgent};
use crate::{
    BurnBackend,
    builders::{
        a2c::{
            agent2::{A2C2BurnAgentBuilder, A2C2CandleAgentBuilder},
            hook2::DefaultA2CHook2Builder,
        },
        agent2::{AgentBuilder2, OnPolicyAgentBuilder2},
        learning_module::OnPolicyLearningModuleType,
        on_policy2::OnPolicyAlgorithmBuilder2,
        sampler2::{Sampler2Builder, SamplerHook2Builder, StepHookBound},
    },
    hooks::a2c::A2CStats,
};

impl<A: Agent2, B, EB: EnvBuilder, SH: SamplerHook2Builder<Env = EB::Env>>
    OnPolicyAlgorithmBuilder2<
        A,
        OnPolicyAgentBuilder2<A2CParams, DefaultA2CHook2Builder, B>,
        EB,
        SH,
    >
where
    OnPolicyAgentBuilder2<A2CParams, DefaultA2CHook2Builder, B>: AgentBuilder2<Agent = A>,
{
    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.agent_builder = self.agent_builder.with_log_progress(log_progress);
        self
    }

    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.agent_builder = self.agent_builder.with_entropy_coeff(entropy_coeff);
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_vf_coeff(vf_coeff);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.agent_builder = self.agent_builder.with_reporter(tx);
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.agent_builder = self.agent_builder.with_gamma(gamma);
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.agent_builder = self.agent_builder.with_lambda(lambda);
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.agent_builder = self.agent_builder.with_sample_size(sample_size);
        self
    }

    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_policy_hidden_layers(policy_hidden_layers);
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.agent_builder = self.agent_builder.with_learning_rate(learning_rate);
        self
    }

    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.agent_builder = self.agent_builder.with_beta1(beta1);
        self
    }

    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.agent_builder = self.agent_builder.with_beta2(beta2);
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.agent_builder = self.agent_builder.with_epsilon(epsilon);
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.agent_builder = self.agent_builder.with_weight_decay(weight_decay);
        self
    }

    pub fn with_joint(mut self, max_grad_norm: Option<f32>, params: ParamsAdamW) -> Self {
        self.agent_builder = self.agent_builder.with_joint(max_grad_norm, params);
        self
    }

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

    pub fn with_value_hidden_layers(mut self, value_hidden_layers: Vec<usize>) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_value_hidden_layers(value_hidden_layers);
        self
    }

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

pub type A2C2CandleAlgorithmBuilder<EB, SH = StepHookBound<<EB as EnvBuilder>::Env>> =
    OnPolicyAlgorithmBuilder2<A2C2CandleAgent, A2C2CandleAgentBuilder, EB, SH>;

impl A2C2CandleAlgorithmBuilder<GymEnvBuilder> {
    pub fn gym<EB: Into<GymEnvBuilder>>(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            Sampler2Builder::new(builder, n_envs),
            A2C2CandleAgentBuilder::new(n_envs),
        )
    }
}

impl<EB: EnvBuilder> A2C2CandleAlgorithmBuilder<EB> {
    pub fn new(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            Sampler2Builder::new(builder, n_envs),
            A2C2CandleAgentBuilder::new(n_envs),
        )
    }
}

pub type A2C2BurnAlgorithmBuilder<EB, SH = StepHookBound<<EB as EnvBuilder>::Env>> =
    OnPolicyAlgorithmBuilder2<A2C2BurnAgent<BurnBackend>, A2C2BurnAgentBuilder, EB, SH>;

impl<EB: EnvBuilder, SH: SamplerHook2Builder<Env = EB::Env>>
    A2C2BurnAlgorithmBuilder<EB, SH>
{
    pub fn with_candle(self, device: candle_core::Device) -> A2C2CandleAlgorithmBuilder<EB, SH> {
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

    pub fn with_burn(self) -> A2C2BurnAlgorithmBuilder<EB, SH> {
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

pub type A2C2AlgorithmBuilder<EB, SH = StepHookBound<<EB as EnvBuilder>::Env>> =
    A2C2CandleAlgorithmBuilder<EB, SH>;

impl<EB: EnvBuilder, SH: SamplerHook2Builder<Env = EB::Env>>
    A2C2CandleAlgorithmBuilder<EB, SH>
{
    pub fn with_candle(self, device: Device) -> A2C2CandleAlgorithmBuilder<EB, SH> {
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

    pub fn with_burn(self) -> A2C2BurnAlgorithmBuilder<EB, SH> {
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
