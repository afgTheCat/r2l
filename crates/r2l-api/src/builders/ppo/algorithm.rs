use std::sync::mpsc::Sender;

use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_core::env::{EnvBuilder, TensorOfEnvBuilder};
use r2l_core::on_policy::algorithm::Agent;
use r2l_gym::GymEnvBuilder;
use r2l_sampler::{StepTrajectoryBound, TrajectoryBound};

use crate::agents::ppo::{BurnPPO, CandlePPO};
use crate::{
    BurnBackend,
    builders::{
        agent::AgentBuilder,
        learning_module::LearningModuleType,
        on_policy::OnPolicyAlgorithmBuilder,
        ppo::agent::{BurnPPOAgentBuilder, CandlePPOAgentBuilder, PPOAgentBuilder},
        sampler::SamplerBuilder,
    },
    hooks::ppo::PPOStats,
};

impl<A, M, EB, BD> OnPolicyAlgorithmBuilder<A, PPOAgentBuilder<M>, EB, BD>
where
    A: Agent,
    EB: EnvBuilder,
    BD: TrajectoryBound<Tensor = TensorOfEnvBuilder<EB>>,
    PPOAgentBuilder<M>: AgentBuilder<Agent = A>,
{
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_total_epochs(total_epochs);
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

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_target_kl(target_kl);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder = self.agent_builder.with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.agent_builder = self.agent_builder.with_reporter(tx);
        self
    }

    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.agent_builder = self.agent_builder.with_clip_range(clip_range);
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

    pub fn with_learning_module_type(mut self, learning_module_type: LearningModuleType) -> Self {
        self.agent_builder = self
            .agent_builder
            .with_learning_module_type(learning_module_type);
        self
    }
}

pub type PPOCandleAlgorithmBuilder<EB, BD = StepTrajectoryBound<TensorOfEnvBuilder<EB>>> =
    OnPolicyAlgorithmBuilder<CandlePPO, CandlePPOAgentBuilder, EB, BD>;

impl PPOCandleAlgorithmBuilder<GymEnvBuilder> {
    pub fn gym<EB: Into<GymEnvBuilder>>(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            SamplerBuilder::new(builder, n_envs),
            CandlePPOAgentBuilder::new(n_envs),
        )
    }
}

impl<EB: EnvBuilder> PPOCandleAlgorithmBuilder<EB> {
    pub fn new(builder: EB, n_envs: usize) -> Self {
        Self::from_sampler_and_agent_builder(
            SamplerBuilder::new(builder, n_envs),
            CandlePPOAgentBuilder::new(n_envs),
        )
    }
}

pub type PPOBurnAlgorithmBuilder<EB, BD = StepTrajectoryBound<TensorOfEnvBuilder<EB>>> =
    OnPolicyAlgorithmBuilder<BurnPPO<BurnBackend>, BurnPPOAgentBuilder, EB, BD>;

impl<EB: EnvBuilder> PPOBurnAlgorithmBuilder<EB> {
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

pub type PPOAlgorithmBuilder<EB, BD = StepTrajectoryBound<TensorOfEnvBuilder<EB>>> =
    PPOCandleAlgorithmBuilder<EB, BD>;

impl<EB: EnvBuilder> PPOCandleAlgorithmBuilder<EB> {
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
