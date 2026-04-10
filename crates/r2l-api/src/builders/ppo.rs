use crate::{
    builders::agent::AgentBuilder,
    hooks::ppo::{PPOStats, StandardPPOHook, StandardPPOHookReporter, TargetKl},
};
use r2l_core::{
    agents::Agent,
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::LearningSchedule,
    sampler::buffer::{StepTrajectoryBound, TrajectoryBound},
};
use std::{marker::PhantomData, sync::mpsc::Sender};

use crate::{
    agents::ppo::{BurnBackend, BurnOrCandlePPO, BurnPPO, CandlePPO},
    builders::agent::{
        PPOAgentBuilder, PPOBurnLearningModuleBuilder, PPOBurnOrCandleLearningModuleBuilder,
        PPOCandleLearningModuleBuilder,
    },
    builders::{on_policy::OnPolicyAlgorightmBuilder, sampler::SamplerBuilder},
};

#[derive(Debug, Clone)]
pub struct StandardPPOHookBuilder {
    normalize_advantage: bool,
    total_epochs: usize,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    target_kl: Option<f32>,
    gradient_clipping: Option<f32>,
    tx: Option<Sender<PPOStats>>,
}

impl Default for StandardPPOHookBuilder {
    fn default() -> Self {
        Self {
            normalize_advantage: true,
            total_epochs: 10,
            entropy_coeff: 0.,
            vf_coeff: None,
            target_kl: None,
            gradient_clipping: None,
            tx: None,
        }
    }
}

impl StandardPPOHookBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.normalize_advantage = normalize_advantage;
        self
    }

    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.total_epochs = total_epochs;
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.entropy_coeff = entropy_coeff;
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.vf_coeff = vf_coeff;
        self
    }

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.target_kl = target_kl;
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.gradient_clipping = gradient_clipping;
        self
    }

    pub fn with_tx(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.tx = tx;
        self
    }

    pub fn normalize_advantage(&self) -> bool {
        self.normalize_advantage
    }

    pub fn total_epochs(&self) -> usize {
        self.total_epochs
    }

    pub fn entropy_coeff(&self) -> f32 {
        self.entropy_coeff
    }

    pub fn vf_coeff(&self) -> Option<f32> {
        self.vf_coeff
    }

    pub fn target_kl(&self) -> Option<f32> {
        self.target_kl
    }

    pub fn gradient_clipping(&self) -> Option<f32> {
        self.gradient_clipping
    }

    pub fn build<T>(self) -> StandardPPOHook<T> {
        StandardPPOHook {
            normalize_advantage: self.normalize_advantage,
            total_epochs: self.total_epochs,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            target_kl: self.target_kl.map(|target| TargetKl {
                target,
                target_exceeded: false,
            }),
            gradient_clipping: self.gradient_clipping,
            current_epoch: 0,
            reporter: self.tx.map(|tx| StandardPPOHookReporter {
                report: PPOStats::default(),
                tx,
            }),
            _lm: PhantomData,
        }
    }
}

impl<A, M, EB, BD> OnPolicyAlgorightmBuilder<A, PPOAgentBuilder<M>, EB, BD>
where
    A: Agent,
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor>,
    PPOAgentBuilder<M>: AgentBuilder<Agent = A>,
{
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
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

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.agent_builder.hook_builder = self.agent_builder.hook_builder.with_target_kl(target_kl);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.agent_builder.hook_builder = self
            .agent_builder
            .hook_builder
            .with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.agent_builder.hook_builder = self.agent_builder.hook_builder.with_tx(tx);
        self
    }

    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.agent_builder.ppo_params.clip_range = clip_range;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.agent_builder.ppo_params.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.agent_builder.ppo_params.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.agent_builder.ppo_params.sample_size = sample_size;
        self
    }
}

pub type PPOCandleAlgorithmBuiler<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<CandlePPO, PPOCandleLearningModuleBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> PPOCandleAlgorithmBuiler<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: PPOCandleLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}

pub type PPOBurnAlgorithmBuiler<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<BurnPPO<BurnBackend>, PPOBurnLearningModuleBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> PPOBurnAlgorithmBuiler<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: PPOBurnLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }
}

pub type PPOAlgorithmBuilder<EB, BD = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>> =
    OnPolicyAlgorightmBuilder<BurnOrCandlePPO, PPOBurnOrCandleLearningModuleBuilder, EB, BD>;

impl<EB: EnvBuilderTrait> PPOAlgorithmBuilder<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        OnPolicyAlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: PPOBurnOrCandleLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        }
    }

    pub fn with_candle(self, device: candle_core::Device) -> PPOCandleAlgorithmBuiler<EB> {
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

    pub fn with_burn(self) -> PPOBurnAlgorithmBuiler<EB> {
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
