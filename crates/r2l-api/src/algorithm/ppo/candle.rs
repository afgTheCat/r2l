use crate::{
    agents::ppo::{CandlePPO, PPOCandleLearningModuleBuilder},
    algorithm::OnPolicyAlgorightmBuilder,
    hooks::ppo::PPOStats,
    sampler::SamplerBuilder,
};
use r2l_core::sampler::Location;
use r2l_core::{
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::LearningSchedule,
    sampler::buffer::{StepTrajectoryBound, TrajectoryBound},
};
use std::sync::mpsc::Sender;

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

impl<EB: EnvBuilderTrait, BD: TrajectoryBound<Tensor = EB::Tensor>>
    PPOCandleAlgorithmBuiler<EB, BD>
{
    pub fn with_bound<BD2: TrajectoryBound<Tensor = EB::Tensor>>(
        self,
        trajectory_bound: BD2,
    ) -> PPOCandleAlgorithmBuiler<EB, BD2> {
        let OnPolicyAlgorightmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            ..
        } = self;
        OnPolicyAlgorightmBuilder {
            sampler_builder: sampler_builder.with_bound(trajectory_bound),
            agent_builder,
            learning_schedule,
        }
    }

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

    pub fn with_location(mut self, location: Location) -> Self {
        self.sampler_builder = self.sampler_builder.with_location(location);
        self
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.learning_schedule = learning_schedule;
        self
    }
}
