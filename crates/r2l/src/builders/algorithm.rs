use std::num::NonZeroUsize;

use r2l_agents::on_policy_algorithms::{
    a2c::{A2C, A2CParams},
    ppo::{PPO, PPOParams},
};
use r2l_core::on_policy::learning_module::OnPolicyLearner;
use serde::{Deserialize, Serialize};

use super::{learner::LearningHookConfig, optimizer::OptimizerConfig};
use crate::{
    A2CSettings, ClipRangeSchedule, PPOSettings,
    hooks::{
        learning::{A2CLearningHook, PPOLearningHook, TargetKl, reporter::RolloutReporter},
        progress::SharedTrainingProgress,
    },
};

/// Concrete algorithm and backend settings selected by an on-policy builder.
#[derive(Serialize, Deserialize)]
pub struct AlgoBackendConfig<A, B> {
    /// Algorithm-specific settings.
    pub algo: A,
    /// Concrete backend configuration.
    pub backend: B,
}

/// PPO-specific epoch, clipping, and early-stopping settings.
#[derive(Serialize, Deserialize)]
pub struct PPOConfig {
    /// Maximum optimization epochs per rollout.
    pub total_epochs: NonZeroUsize,
    /// Policy-ratio clipping schedule.
    pub clip_range_schedule: ClipRangeSchedule,
    /// Optional KL-divergence threshold for early stopping.
    pub target_kl: Option<f32>,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            total_epochs: NonZeroUsize::new(10).unwrap(),
            clip_range_schedule: ClipRangeSchedule::Constant(0.2),
            target_kl: None,
        }
    }
}

/// Selects A2C, which uses only the shared on-policy learning settings.
#[derive(Default, Serialize, Deserialize)]
pub struct A2CConfig;

/// Learning settings shared by PPO and A2C.
#[derive(Serialize, Deserialize)]
pub struct AlgoConfig {
    /// Discount factor used for returns.
    pub gamma: f32,
    /// Generalized advantage estimation decay factor.
    pub lambda: f32,
    /// Number of samples per minibatch.
    pub sample_size: NonZeroUsize,
    /// Whether to print rollout statistics.
    pub log_progress: bool,
}

impl Default for AlgoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.98,
            lambda: 0.8,
            sample_size: NonZeroUsize::new(64).unwrap(),
            log_progress: true,
        }
    }
}

impl AlgoConfig {
    pub(crate) fn build_ppo<M: OnPolicyLearner>(
        &self,
        config: &PPOConfig,
        learner: M,
        learning_hook: &LearningHookConfig,
        optimizer: &OptimizerConfig,
        progress: SharedTrainingProgress,
        n_envs: NonZeroUsize,
    ) -> PPO<M, PPOLearningHook> {
        let params = PPOParams {
            clip_range: config.clip_range_schedule.initial_value(),
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size.get(),
        };
        let settings = PPOSettings {
            total_epochs: config.total_epochs,
            current_epoch: 0,
            clip_range_schedule: config.clip_range_schedule,
            target_kl: config.target_kl.map(|target| TargetKl {
                target,
                target_exceeded: false,
            }),
            reporter: RolloutReporter::new(None, self.log_progress, n_envs),
        };
        PPO {
            lm: learner,
            hooks: learning_hook.build(settings, optimizer, progress),
            params,
        }
    }

    pub(crate) fn build_a2c<M: OnPolicyLearner>(
        &self,
        learner: M,
        learning_hook: &LearningHookConfig,
        optimizer: &OptimizerConfig,
        progress: SharedTrainingProgress,
        n_envs: NonZeroUsize,
    ) -> A2C<M, A2CLearningHook> {
        let params = A2CParams {
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size.get(),
        };
        let settings = A2CSettings {
            reporter: RolloutReporter::new(None, self.log_progress, n_envs),
        };
        A2C {
            lm: learner,
            hooks: learning_hook.build(settings, optimizer, progress),
            params,
        }
    }
}
