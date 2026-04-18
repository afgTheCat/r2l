use std::sync::mpsc::Sender;

use candle_core::Device;
use r2l_core::{env::EnvBuilderTrait, on_policy::algorithm::Agent};
use r2l_sampler::{StepTrajectoryBound, TrajectoryBound};

use crate::{
    BurnBackend,
    agents::a2c::{BurnA2C, CandleA2C},
    builders::{
        a2c::agent::{A2CAgentBuilder, A2CBurnAgentBuilder, A2CCandleAgentBuilder},
        agent::AgentBuilder,
        learning_module::LearningModuleType,
        on_policy::OnPolicyAlgorightmBuilder,
        sampler::SamplerBuilder,
    },
    hooks::{a2c::A2CStats, on_policy::LearningSchedule},
};

impl<A, M, EB, BD> OnPolicyAlgorightmBuilder<A, A2CAgentBuilder<M>, EB, BD>
where
    A: Agent,
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor>,
    A2CAgentBuilder<M>: AgentBuilder<Agent = A>,
{
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_entropy_coeff(entropy_coeff);
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.agent_builder.hook_builder = self.agent_builder.hook_builder.with_vf_coeff(vf_coeff);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.agent_builder.hook_builder = self.agent_builder.hook_builder.with_tx(tx);
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.agent_builder.a2c_params.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.agent_builder.a2c_params.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.agent_builder.a2c_params.sample_size = sample_size;
        self
    }

    pub fn with_policy_hidden_layers(mut self, policy_hidden_layers: Vec<usize>) -> Self {
        self.agent_builder
            .learning_module_builder
            .policy_hidden_layers = policy_hidden_layers;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.agent_builder.learning_module_builder.params.lr = learning_rate;
        self
    }

    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.agent_builder.learning_module_builder.params.beta1 = beta1;
        self
    }

    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.agent_builder.learning_module_builder.params.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.agent_builder.learning_module_builder.params.eps = epsilon;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.agent_builder
            .learning_module_builder
            .params
            .weight_decay = weight_decay;
        self
    }

    pub fn with_joint(
        mut self,
        value_hidden_layers: Vec<usize>,
        max_grad_norm: Option<f32>,
    ) -> Self {
        self.agent_builder
            .learning_module_builder
            .learning_module_type = LearningModuleType::Joint {
            value_hidden_layers,
            max_grad_norm,
        };
        self
    }

    pub fn with_split(
        mut self,
        value_hidden_layers: Vec<usize>,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    ) -> Self {
        self.agent_builder
            .learning_module_builder
            .learning_module_type = LearningModuleType::Split {
            value_hidden_layers,
            policy_max_grad_norm,
            value_max_grad_norm,
        };
        self
    }

    pub fn with_learning_module_type(mut self, learning_module_type: LearningModuleType) -> Self {
        self.agent_builder
            .learning_module_builder
            .learning_module_type = learning_module_type;
        self
    }
}

pub type A2CBurnAlgorithmBuilder<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<BurnA2C<BurnBackend>, A2CBurnAgentBuilder, EB, BD>;

pub type A2CCandleAlgorithmBuilder<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<CandleA2C, A2CCandleAgentBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> A2CCandleAlgorithmBuilder<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        let agent_builder = A2CCandleAgentBuilder::new(n_envs);
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder,
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}

pub type A2CAlgorithmBuilder<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    A2CCandleAlgorithmBuilder<EB, BD>;

impl<EB: EnvBuilderTrait> A2CAlgorithmBuilder<EB> {
    pub fn with_candle(self, device: Device) -> A2CCandleAlgorithmBuilder<EB> {
        let OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_candle(device),
        }
    }

    pub fn with_burn(self) -> A2CBurnAlgorithmBuilder<EB> {
        let OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_burn(),
        }
    }
}

impl<EB: EnvBuilderTrait> A2CBurnAlgorithmBuilder<EB> {
    pub fn with_candle(self, device: Device) -> A2CCandleAlgorithmBuilder<EB> {
        let OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_candle(device),
        }
    }

    pub fn with_burn(self) -> A2CBurnAlgorithmBuilder<EB> {
        let OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder,
        } = self;
        OnPolicyAlgorightmBuilder {
            sampler_builder,
            learning_schedule,
            agent_builder: agent_builder.with_burn(),
        }
    }
}
