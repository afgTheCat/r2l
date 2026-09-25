//! Shared PPO and A2C learning behavior for Candle and Burn.

pub(crate) mod reporter;
pub(crate) mod stats;

use std::marker::PhantomData;

use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    a2c::{A2CBatchData, A2CHook, A2CParams},
    ppo::{PPOBatchData, PPOHook, PPOParams},
};
use r2l_core::{
    HookResult, buffers::TrajectoryBatch, error::Result,
    on_policy::learning_module::OnPolicyLearner, tensor::R2lTensor,
};
use r2l_distributions::learning_modules::burn_lm::{
    BurnPolicy, PolicyValueLearner as BurnLearner, PolicyValueLosses as BurnLosses,
};
use r2l_distributions::learning_modules::candle_lm::{
    PolicyValueLearner as CandleLearner, PolicyValueLosses as CandleLosses,
};

use self::{
    reporter::RolloutReporter,
    stats::{A2CMinibatchStats, A2CRolloutStats, PPORolloutStats},
};
use super::progress::SharedTrainingProgress;

/// Learning-rate policy applied to shared collection progress.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule {
    /// Keep the learning rate fixed throughout training.
    Constant(f64),
    /// Decay the initial learning rate to zero, including the current collection in progress.
    /// The final learning pass uses zero learning rate, including a one-rollout run.
    Linear(f64),
}

impl LearningRateSchedule {
    /// Returns the learning rate for the remaining training fraction.
    ///
    /// # Arguments
    ///
    /// * `progress_remaining` - Remaining fraction, clamped to `[0, 1]` for linear decay.
    pub(crate) fn value(self, progress_remaining: f64) -> f64 {
        match self {
            Self::Constant(learning_rate) => learning_rate,
            Self::Linear(initial_learning_rate) => {
                initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
            }
        }
    }
}

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
    pub(crate) total_epochs: usize,
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
        self.current_epoch == self.total_epochs || target_exceeded
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
    pub(crate) vf_coeff: Option<f32>,
    pub(crate) gradient_clipping: Option<f32>,
    pub(crate) progress: SharedTrainingProgress,
    pub(crate) learning_rate_schedule: LearningRateSchedule,
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
        module.set_learning_rate(self.learning_rate_schedule.value(progress));
        if self.normalize_advantage {
            advantages.normalize();
        }
        progress
    }
}

// Backend-specific operations are shared by both algorithm adapters below.
impl<B: AutodiffBackend, P: BurnPolicy<B>, A> LearningHook<BurnLearner<B, P>, A> {
    fn prepare(&self, module: &mut BurnLearner<B, P>, advantages: &mut Advantages) -> f64 {
        let progress = self.prepare_learning(module, advantages);
        if let Some(max_grad_norm) = self.gradient_clipping {
            module.set_grad_clipping(GradientClipping::Norm(max_grad_norm));
        }
        progress
    }

    fn process_batch(
        &self,
        module: &mut BurnLearner<B, P>,
        losses: &mut BurnLosses<B>,
        observations: &burn::Tensor<B, 2>,
        collect_stats: bool,
    ) -> Result<Option<A2CMinibatchStats>> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy_loss =
            module.policy().entropy(observations.clone())?.neg() * self.entropy_coeff;
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
            losses.add_entropy_loss(entropy_loss);
        }
        Ok(stats)
    }
}

impl<P: r2l_core::models::Policy<Tensor = Tensor> + Clone, A> LearningHook<CandleLearner<P>, A> {
    fn prepare(&self, module: &mut CandleLearner<P>, advantages: &mut Advantages) -> Result<f64> {
        let progress = self.prepare_learning(module, advantages);
        module.set_grad_clipping(self.gradient_clipping)?;
        Ok(progress)
    }

    fn process_batch(
        &self,
        module: &mut CandleLearner<P>,
        losses: &mut CandleLosses,
        observations: &Tensor,
        collect_stats: bool,
    ) -> Result<Option<A2CMinibatchStats>> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(observations.clone())?;
        let entropy_loss =
            (Tensor::full(self.entropy_coeff, (), entropy.device())? * entropy.neg()?)?;
        let stats = if collect_stats {
            Some(A2CMinibatchStats {
                policy_loss: losses.policy_loss.to_scalar()?,
                entropy_loss: entropy_loss.to_scalar()?,
                value_loss: losses.value_loss.to_scalar()?,
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

impl<B: AutodiffBackend, P: BurnPolicy<B>> A2CHook<BurnLearner<B, P>>
    for LearningHook<BurnLearner<B, P>, A2CSettings>
{
    fn before_learning_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 2>>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnLearner<B, P>,
        _batches: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        self.prepare(module, advantages);
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnLearner<B, P>,
        losses: &mut BurnLosses<B>,
        data: &A2CBatchData<burn::Tensor<B, 2>>,
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

    fn after_learning_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 2>>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnLearner<B, P>,
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

impl<P: r2l_core::models::Policy<Tensor = Tensor> + Clone> A2CHook<CandleLearner<P>>
    for LearningHook<CandleLearner<P>, A2CSettings>
{
    fn before_learning_hook<T: TrajectoryBatch<Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandleLearner<P>,
        _batches: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        self.prepare(module, advantages)?;
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandleLearner<P>,
        losses: &mut CandleLosses,
        data: &A2CBatchData<Tensor>,
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

    fn after_learning_hook<T: TrajectoryBatch<Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandleLearner<P>,
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

impl<B: AutodiffBackend, P: BurnPolicy<B>> PPOHook<BurnLearner<B, P>>
    for LearningHook<BurnLearner<B, P>, PPOSettings>
{
    fn before_learning_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 2>>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnLearner<B, P>,
        _batches: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        let progress = self.prepare(module, advantages);
        self.algorithm.begin_learning(params, progress);
        Ok(HookResult::Continue)
    }

    fn rollout_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 2>>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnLearner<B, P>,
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
        module: &mut BurnLearner<B, P>,
        losses: &mut BurnLosses<B>,
        data: &PPOBatchData<burn::Tensor<B, 2>>,
    ) -> Result<HookResult> {
        let stats = self.process_batch(
            module,
            losses,
            &data.observations,
            self.algorithm.reporter.is_enabled(),
        )?;
        let ratio = data.ratio.to_vec()?;
        let log_ratio = data.logp_diff.to_vec()?;
        let approx_kl = ratio
            .iter()
            .zip(&log_ratio)
            .map(|(ratio, log_ratio)| (ratio - 1.) - log_ratio)
            .sum::<f32>()
            / ratio.len() as f32;
        let clip_fraction = if stats.is_some() {
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

impl<P: r2l_core::models::Policy<Tensor = Tensor> + Clone> PPOHook<CandleLearner<P>>
    for LearningHook<CandleLearner<P>, PPOSettings>
{
    fn before_learning_hook<T: TrajectoryBatch<Tensor>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut CandleLearner<P>,
        _batches: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        let progress = self.prepare(module, advantages)?;
        self.algorithm.begin_learning(params, progress);
        Ok(HookResult::Continue)
    }

    fn rollout_hook<T: TrajectoryBatch<Tensor>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut CandleLearner<P>,
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
        module: &mut CandleLearner<P>,
        losses: &mut CandleLosses,
        data: &PPOBatchData<Tensor>,
    ) -> Result<HookResult> {
        let stats = self.process_batch(
            module,
            losses,
            &data.observations,
            self.algorithm.reporter.is_enabled(),
        )?;
        let ratio = data.ratio.detach();
        let log_ratio = data.logp_diff.detach();
        let approx_kl = ratio
            .sub(&Tensor::ones_like(&ratio)?)?
            .sub(&log_ratio)?
            .mean_all()?
            .to_scalar::<f32>()?;
        let clip_fraction = if stats.is_some() {
            (&data.ratio - 1.)?
                .abs()?
                .gt(params.clip_range)?
                .to_dtype(candle_core::DType::F32)?
                .mean_all()?
                .to_scalar::<f32>()?
        } else {
            0.
        };
        self.algorithm
            .reporter
            .record_batch(stats, clip_fraction, approx_kl);
        Ok(self.algorithm.check_kl(approx_kl))
    }
}
