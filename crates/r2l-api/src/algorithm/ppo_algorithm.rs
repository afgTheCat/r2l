use crate::{
    agents::{AgentBuilder, PPOCandleLearningModuleBuilder, candle_ppo2::CandlePPO},
    algorithm::AlgorightmBuilder,
    builders::distribution::ActionSpaceType,
    hooks::ppo::PPOStats,
    sampler::SamplerBuilder,
};
use r2l_core::sampler::Location;
use r2l_core::{
    env::Space,
    env_builder::EnvBuilderTrait,
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
    sampler::{
        FinalSampler,
        buffer::{StepTrajectoryBound, TrajectoryBound},
    },
};
use std::sync::mpsc::Sender;

pub struct PPOAlgorithmBuiler<
    EB: EnvBuilderTrait,
    BD: TrajectoryBound<Tensor = EB::Tensor> = StepTrajectoryBound<<EB as EnvBuilderTrait>::Tensor>,
>(AlgorightmBuilder<CandlePPO, PPOCandleLearningModuleBuilder, EB, BD>);

impl<EB: EnvBuilderTrait> PPOAlgorithmBuiler<EB> {
    pub fn new<B: Into<EB>>(builder: B, n_envs: usize) -> Self {
        Self(AlgorightmBuilder {
            sampler_builder: SamplerBuilder::new(builder, n_envs),
            agent_builder: PPOCandleLearningModuleBuilder::default(),
            learning_schedule: LearningSchedule::RolloutBound {
                total_rollouts: 300,
                current_rollout: 0,
            },
        })
    }
}

impl<EB: EnvBuilderTrait, BD: TrajectoryBound<Tensor = EB::Tensor>> PPOAlgorithmBuiler<EB, BD> {
    pub fn with_bound<BD2: TrajectoryBound<Tensor = EB::Tensor>>(
        self,
        trajectory_bound: BD2,
    ) -> PPOAlgorithmBuiler<EB, BD2> {
        let AlgorightmBuilder {
            sampler_builder,
            agent_builder,
            learning_schedule,
            ..
        } = self.0;
        PPOAlgorithmBuiler(AlgorightmBuilder {
            sampler_builder: sampler_builder.with_bound(trajectory_bound),
            agent_builder,
            learning_schedule,
        })
    }

    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.0.agent_builder.hook_builder = self
            .0
            .agent_builder
            .hook_builder
            .with_normalize_advantage(normalize_advantage);
        self
    }

    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.0.agent_builder.hook_builder = self
            .0
            .agent_builder
            .hook_builder
            .with_total_epochs(total_epochs);
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.0.agent_builder.hook_builder = self
            .0
            .agent_builder
            .hook_builder
            .with_entropy_coeff(entropy_coeff);
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.0.agent_builder.hook_builder =
            self.0.agent_builder.hook_builder.with_vf_coeff(vf_coeff);
        self
    }

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.0.agent_builder.hook_builder =
            self.0.agent_builder.hook_builder.with_target_kl(target_kl);
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.0.agent_builder.hook_builder = self
            .0
            .agent_builder
            .hook_builder
            .with_gradient_clipping(gradient_clipping);
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.0.agent_builder.hook_builder = self.0.agent_builder.hook_builder.with_tx(tx);
        self
    }

    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.0.agent_builder.ppo_params.clip_range = clip_range;
        self
    }

    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.0.agent_builder.ppo_params.gamma = gamma;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.0.agent_builder.ppo_params.lambda = lambda;
        self
    }

    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.0.agent_builder.ppo_params.sample_size = sample_size;
        self
    }

    pub fn with_location(mut self, location: Location) -> Self {
        self.0.sampler_builder = self.0.sampler_builder.with_location(location);
        self
    }

    pub fn with_learning_schedule(mut self, learning_schedule: LearningSchedule) -> Self {
        self.0.learning_schedule = learning_schedule;
        self
    }

    // TODO: too much. Also not generic enough
    pub fn build(
        self,
    ) -> anyhow::Result<
        OnPolicyAlgorithm<
            CandlePPO,
            FinalSampler<EB::Env, BD>,
            DefaultOnPolicyAlgorightmsHooks<CandlePPO, FinalSampler<EB::Env, BD>>,
        >,
    > {
        let env_description = self.0.sampler_builder.env_builder.env_description()?;
        let sampler = self.0.sampler_builder.build();
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space = match env_description.action_space {
            Space::Discrete(_) => ActionSpaceType::Discrete,
            Space::Continous { .. } => ActionSpaceType::Continous,
        };
        let agent = self
            .0
            .agent_builder
            .build(observation_size, action_size, action_space)?;
        let hooks = DefaultOnPolicyAlgorightmsHooks::new(LearningSchedule::RolloutBound {
            total_rollouts: 300,
            current_rollout: 0,
        });
        Ok(OnPolicyAlgorithm {
            sampler,
            agent,
            hooks,
        })
    }
}
