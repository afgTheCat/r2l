//! Shared PPO and A2C learning behavior for Candle and Burn.

pub(crate) mod reporter;
pub(crate) mod stats;

use std::{marker::PhantomData, num::NonZeroUsize};

use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    a2c::{A2CBatchData, A2CHook, A2CParams},
    ppo::{PPOBatchData, PPOHook, PPOParams},
};
use r2l_core::{
    HookResult, buffers::TrajectoryBatch, error::Result, models::Policy,
    on_policy::learning_module::OnPolicyLearner, on_policy::losses::PolicyValueLosses,
    tensor::R2lTensor,
};

use self::{
    reporter::RolloutReporter,
    stats::{A2CMinibatchStats, A2CRolloutStats, PPORolloutStats},
};
use super::progress::SharedTrainingProgress;
use crate::LearningRateSchedule;

/// Policy-ratio clipping range applied over the progress of PPO training.
#[derive(Debug, Clone, Copy)]
pub enum ClipRangeSchedule {
    /// Keep the clipping range fixed throughout training.
    Constant(f32),
    /// Decay the initial clipping range linearly to zero.
    Linear(f32),
}

impl ClipRangeSchedule {
    pub(crate) fn initial_value(self) -> f32 {
        match self {
            Self::Constant(clip_range) | Self::Linear(clip_range) => clip_range,
        }
    }
}

pub(crate) struct TargetKl {
    pub target: f32,
    pub target_exceeded: bool,
}

impl TargetKl {
    pub(crate) fn target_kl_exceeded(&mut self) -> bool {
        std::mem::take(&mut self.target_exceeded)
    }
}

/// Reporting state specific to A2C learning.
pub struct A2CSettings {
    pub(crate) reporter: RolloutReporter<A2CRolloutStats>,
}

/// Epoch, clipping, KL, and reporting state specific to PPO learning.
pub struct PPOSettings {
    pub(crate) total_epochs: NonZeroUsize,
    pub(crate) current_epoch: usize,
    pub(crate) clip_range_schedule: ClipRangeSchedule,
    pub(crate) target_kl: Option<TargetKl>,
    pub(crate) reporter: RolloutReporter<PPORolloutStats>,
}

impl PPOSettings {
    fn begin_learning(&mut self, params: &mut PPOParams, progress: f64) {
        self.current_epoch = 0;
        params.clip_range = match self.clip_range_schedule {
            ClipRangeSchedule::Constant(value) => value,
            ClipRangeSchedule::Linear(initial) => initial * progress as f32,
        };
    }

    fn finish_epoch(&mut self) -> bool {
        self.current_epoch += 1;
        let target_exceeded = self
            .target_kl
            .as_mut()
            .is_some_and(TargetKl::target_kl_exceeded);
        self.current_epoch == self.total_epochs.get() || target_exceeded
    }

    fn check_kl(&mut self, approx_kl: f32) -> HookResult {
        if let Some(target_kl) = &mut self.target_kl
            && approx_kl > 1.5 * target_kl.target
        {
            target_kl.target_exceeded = true;
            HookResult::Break
        } else {
            HookResult::Continue
        }
    }
}

/// Shared learning configuration with algorithm-specific state in `A`.
pub struct LearningHook<M, A> {
    pub(crate) normalize_advantage: bool,
    pub(crate) entropy_coeff: f32,
    pub(crate) vf_coeff: f32,
    pub(crate) progress: SharedTrainingProgress,
    pub(crate) policy_learning_rate_schedule: LearningRateSchedule,
    pub(crate) value_learning_rate_schedule: LearningRateSchedule,
    pub(crate) algorithm: A,
    pub(crate) _lm: PhantomData<M>,
}

/// Shared hook specialized for PPO learning.
pub type PPOLearningHook<M = ()> = LearningHook<M, PPOSettings>;
/// Shared hook specialized for A2C learning.
pub type A2CLearningHook<M = ()> = LearningHook<M, A2CSettings>;

impl<M: OnPolicyLearner, A> LearningHook<M, A> {
    fn prepare_learning(&self, module: &mut M, advantages: &mut Advantages) -> f64 {
        let progress = self.progress.borrow().progress_remaining();
        module.set_learning_rates(
            self.policy_learning_rate_schedule.value(progress),
            self.value_learning_rate_schedule.value(progress),
        );
        if self.normalize_advantage {
            advantages.normalize();
        }
        progress
    }

    fn process_batch(
        &self,
        module: &mut M,
        losses: &mut PolicyValueLosses<M::Tensor>,
        observations: &M::Tensor,
        collect_stats: bool,
    ) -> Result<Option<A2CMinibatchStats>> {
        losses.set_vf_coeff(self.vf_coeff);
        if self.entropy_coeff == 0. && !collect_stats {
            return Ok(None);
        }
        let entropy_loss = module
            .policy()
            .entropy(observations.clone())?
            .mul_scalar(-self.entropy_coeff)?;
        let stats = if collect_stats {
            Some(A2CMinibatchStats {
                policy_loss: losses.policy_loss.to_vec()?[0],
                entropy_loss: entropy_loss.to_vec()?[0],
                value_loss: losses.value_loss.to_vec()?[0],
            })
        } else {
            None
        };
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(&entropy_loss)?;
        }
        Ok(stats)
    }
}

impl<M: OnPolicyLearner> A2CHook<M> for LearningHook<M, A2CSettings> {
    fn before_learning_hook<T: TrajectoryBatch<M::Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut M,
        _batches: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        self.prepare_learning(module, advantages);
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut M,
        losses: &mut PolicyValueLosses<M::Tensor>,
        data: &A2CBatchData<M::Tensor>,
    ) -> Result<HookResult> {
        let stats = self.process_batch(
            module,
            losses,
            &data.observations,
            self.algorithm.reporter.is_enabled(),
        )?;
        self.algorithm.reporter.record_batch(stats);
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<T: TrajectoryBatch<M::Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut M,
        batches: &[T],
    ) -> Result<HookResult> {
        if self.algorithm.reporter.is_enabled() {
            let (completed_rollouts, total_rollouts) = {
                let progress = self.progress.borrow();
                (progress.completed_rollouts(), progress.total_rollouts())
            };
            self.algorithm.reporter.report(
                batches,
                completed_rollouts,
                total_rollouts,
                module.policy().std()?,
                module.policy_learning_rate(),
            )?;
        }
        Ok(HookResult::Continue)
    }
}

impl<M: OnPolicyLearner> PPOHook<M> for LearningHook<M, PPOSettings> {
    fn before_learning_hook<T: TrajectoryBatch<M::Tensor>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut M,
        _batches: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        let progress = self.prepare_learning(module, advantages);
        self.algorithm.begin_learning(params, progress);
        Ok(HookResult::Continue)
    }

    fn rollout_hook<T: TrajectoryBatch<M::Tensor>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut M,
        batches: &[T],
    ) -> Result<HookResult> {
        if !self.algorithm.finish_epoch() {
            return Ok(HookResult::Continue);
        }
        if self.algorithm.reporter.is_enabled() {
            let (completed_rollouts, total_rollouts) = {
                let progress = self.progress.borrow();
                (progress.completed_rollouts(), progress.total_rollouts())
            };
            self.algorithm.reporter.report(
                batches,
                completed_rollouts,
                total_rollouts,
                module.policy().std()?,
                module.policy_learning_rate(),
                params.clip_range,
            )?;
        }
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        params: &mut PPOParams,
        module: &mut M,
        losses: &mut PolicyValueLosses<M::Tensor>,
        data: &PPOBatchData<M::Tensor>,
    ) -> Result<HookResult> {
        let stats = self.process_batch(
            module,
            losses,
            &data.observations,
            self.algorithm.reporter.is_enabled(),
        )?;
        if stats.is_none() && self.algorithm.target_kl.is_none() {
            return Ok(HookResult::Continue);
        }
        let approx_kl = data
            .ratio
            .detach()
            .add_scalar(-1.)?
            .sub(&data.logp_diff.detach())?
            .mean()?
            .to_vec()?[0];
        let clip_fraction = if stats.is_some() {
            let ratio = data.ratio.to_vec()?;
            ratio
                .iter()
                .filter(|value| (**value - 1.).abs() > params.clip_range)
                .count() as f32
                / ratio.len() as f32
        } else {
            0.
        };
        self.algorithm
            .reporter
            .record_batch(stats, clip_fraction, approx_kl);
        Ok(self.algorithm.check_kl(approx_kl))
    }
}
