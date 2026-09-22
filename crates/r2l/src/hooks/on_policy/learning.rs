//! Shared PPO and A2C learning behavior for Candle and Burn.

use std::{fmt::Display, marker::PhantomData, sync::mpsc::Sender};

use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    a2c::{A2CBatchData, A2CHook, A2CParams},
    ppo::{PPOBatchData, PPOHook, PPOParams},
};
use r2l_burn::learning_module::{
    BurnPolicy, PolicyValueLearner as BurnLearner, PolicyValueLosses as BurnLosses,
};
use r2l_candle::learning_module::{
    CandlePolicy, PolicyValueLearner as CandleLearner, PolicyValueLosses as CandleLosses,
};
use r2l_core::{
    HookResult,
    buffers::TrajectoryBatch,
    error::{Error, ResourceInterrupted, Result},
    on_policy::learning_module::OnPolicyLearner,
    tensor::R2lTensor,
};

use super::{LearningRateSchedule, coordinator::SharedCoordinator};
use crate::hooks::stats::{
    A2CMinibatchStats, A2CRolloutStats, ClipRangeSchedule, PPOMinibatchStats, PPORolloutStats,
};

/// Shared learning configuration with algorithm-specific state in `A`.
pub struct LearningHook<M, A> {
    pub(crate) normalize_advantage: bool,
    pub(crate) entropy_coeff: f32,
    pub(crate) vf_coeff: Option<f32>,
    pub(crate) gradient_clipping: Option<f32>,
    pub(crate) coordinator: SharedCoordinator,
    pub(crate) learning_rate_schedule: Option<LearningRateSchedule>,
    pub(crate) algorithm: A,
    pub(crate) _lm: PhantomData<M>,
}

/// Reporting state specific to A2C learning.
pub struct A2CSettings {
    pub(crate) reporter: Option<RolloutReporter<A2CRolloutStats>>,
}

/// Epoch, clipping, KL, and reporting state specific to PPO learning.
pub struct PPOSettings {
    pub(crate) total_epochs: usize,
    pub(crate) current_epoch: usize,
    pub(crate) clip_range_schedule: ClipRangeSchedule,
    pub(crate) target_kl: Option<TargetKl>,
    pub(crate) reporter: Option<RolloutReporter<PPORolloutStats>>,
}

/// Shared hook specialized for PPO learning.
pub type PPOLearningHook<M = ()> = LearningHook<M, PPOSettings>;
/// Shared hook specialized for A2C learning.
pub type A2CLearningHook<M = ()> = LearningHook<M, A2CSettings>;

pub(crate) struct TargetKl {
    pub target: f32,
    pub target_exceeded: bool,
}

impl TargetKl {
    pub(crate) fn target_kl_exceeded(&mut self) -> bool {
        std::mem::take(&mut self.target_exceeded)
    }
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

/// Shared reward tracking and delivery, retaining the existing statistics payloads.
pub(crate) struct RolloutReporter<R> {
    report: R,
    tx: Option<Sender<R>>,
    log_progress: bool,
    unfinished_episode_rewards: Vec<f32>,
    latest_average_reward: f32,
}

impl<R: Default + Display> RolloutReporter<R> {
    /// Creates a reporter when logging or channel delivery is enabled.
    ///
    /// # Arguments
    ///
    /// * `tx` - Optional channel receiving each rollout's statistics.
    /// * `log_progress` - Whether to print rollout statistics.
    /// * `n_envs` - Number of environment reward streams to track.
    pub(crate) fn new(tx: Option<Sender<R>>, log_progress: bool, n_envs: usize) -> Option<Self> {
        (tx.is_some() || log_progress).then(|| Self {
            report: R::default(),
            tx,
            log_progress,
            unfinished_episode_rewards: vec![0.; n_envs],
            latest_average_reward: 0.,
        })
    }

    fn update_average_reward<T: R2lTensor, B: TrajectoryBatch<T>>(&mut self, batches: &[B]) {
        let mut completed_episode_rewards = vec![];
        for (running_reward, batch) in self.unfinished_episode_rewards.iter_mut().zip(batches) {
            for (reward, done) in batch.rewards().iter().copied().zip(
                batch
                    .terminated()
                    .iter()
                    .zip(batch.truncated())
                    .map(|(terminated, truncated)| *terminated || *truncated),
            ) {
                *running_reward += reward;
                if done {
                    completed_episode_rewards.push(*running_reward);
                    *running_reward = 0.;
                }
            }
        }
        if !completed_episode_rewards.is_empty() {
            self.latest_average_reward = completed_episode_rewards.iter().sum::<f32>()
                / completed_episode_rewards.len() as f32;
        }
    }

    fn send_report(&mut self) -> Result<()> {
        let report = std::mem::take(&mut self.report);
        if self.log_progress {
            println!("{report}");
        }
        if let Some(tx) = &self.tx {
            tx.send(report).map_err(|error| {
                Error::ResourceInterrupted(ResourceInterrupted {
                    resource: "on-policy rollout reporter".into(),
                    details: error.to_string(),
                })
            })?;
        }
        Ok(())
    }
}

impl<M: OnPolicyLearner, A> LearningHook<M, A> {
    fn prepare_learning(&self, module: &mut M, advantages: &mut Advantages) -> f64 {
        let progress = self.coordinator.borrow().progress_remaining();
        if let Some(schedule) = self.learning_rate_schedule {
            module.set_learning_rate(schedule.value(progress));
        }
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
        observations: &[burn::Tensor<B, 1>],
        collect_stats: bool,
    ) -> Result<Option<A2CMinibatchStats>> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy_loss = module.policy().entropy(observations)?.neg() * self.entropy_coeff;
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

impl<P: CandlePolicy, A> LearningHook<CandleLearner<P>, A> {
    fn prepare(&self, module: &mut CandleLearner<P>, advantages: &mut Advantages) -> f64 {
        let progress = self.prepare_learning(module, advantages);
        module.set_grad_clipping(self.gradient_clipping);
        progress
    }

    fn process_batch(
        &self,
        module: &mut CandleLearner<P>,
        losses: &mut CandleLosses,
        observations: &[Tensor],
        collect_stats: bool,
    ) -> Result<Option<A2CMinibatchStats>> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(observations)?;
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

impl<M> LearningHook<M, A2CSettings> {
    fn record_batch(&mut self, stats: Option<A2CMinibatchStats>) {
        if let (Some(reporter), Some(stats)) = (&mut self.algorithm.reporter, stats) {
            reporter.report.minibatch_stats.push(stats);
        }
    }

    fn report<T: R2lTensor, B: TrajectoryBatch<T>>(
        &mut self,
        batches: &[B],
        std: Option<f32>,
        learning_rate: f64,
    ) -> Result<()> {
        if let Some(reporter) = &mut self.algorithm.reporter {
            reporter.update_average_reward(batches);
            let coordinator = self.coordinator.borrow();
            reporter.report.rollout_idx = coordinator.completed_rollouts();
            reporter.report.total_rollouts = coordinator.total_rollouts();
            drop(coordinator);
            reporter.report.average_reward = reporter.latest_average_reward;
            reporter.report.std = std;
            reporter.report.learning_rate = learning_rate;
            reporter.send_report()?;
        }
        Ok(())
    }
}

impl<M> LearningHook<M, PPOSettings> {
    fn record_batch(
        &mut self,
        stats: Option<A2CMinibatchStats>,
        clip_fraction: f32,
        approx_kl: f32,
    ) {
        if let (Some(reporter), Some(stats)) = (&mut self.algorithm.reporter, stats) {
            reporter.report.minibatch_stats.push(PPOMinibatchStats {
                policy_loss: stats.policy_loss,
                entropy_loss: stats.entropy_loss,
                value_loss: stats.value_loss,
                clip_fraction,
                approx_kl,
            });
        }
    }

    fn report<T: R2lTensor, B: TrajectoryBatch<T>>(
        &mut self,
        batches: &[B],
        std: Option<f32>,
        learning_rate: f64,
        clip_range: f32,
    ) -> Result<()> {
        if let Some(reporter) = &mut self.algorithm.reporter {
            reporter.update_average_reward(batches);
            let coordinator = self.coordinator.borrow();
            reporter.report.rollout_idx = coordinator.completed_rollouts();
            reporter.report.total_rollouts = coordinator.total_rollouts();
            drop(coordinator);
            reporter.report.average_reward = reporter.latest_average_reward;
            reporter.report.std = std;
            reporter.report.learning_rate = learning_rate;
            reporter.report.clip_range = clip_range;
            reporter.send_report()?;
        }
        Ok(())
    }
}

impl<B: AutodiffBackend, P: BurnPolicy<B>> A2CHook<BurnLearner<B, P>>
    for LearningHook<BurnLearner<B, P>, A2CSettings>
{
    fn before_learning_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 1>>>(
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
        data: &A2CBatchData<burn::Tensor<B, 1>>,
    ) -> Result<HookResult> {
        let stats = self.process_batch(
            module,
            losses,
            &data.observations,
            self.algorithm.reporter.is_some(),
        )?;
        self.record_batch(stats);
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 1>>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnLearner<B, P>,
        batches: &[T],
    ) -> Result<HookResult> {
        if self.algorithm.reporter.is_some() {
            self.report(
                batches,
                module.policy().std()?,
                module.policy_learning_rate(),
            )?;
        }
        Ok(HookResult::Continue)
    }
}

impl<P: CandlePolicy> A2CHook<CandleLearner<P>> for LearningHook<CandleLearner<P>, A2CSettings> {
    fn before_learning_hook<T: TrajectoryBatch<Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandleLearner<P>,
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
        module: &mut CandleLearner<P>,
        losses: &mut CandleLosses,
        data: &A2CBatchData<Tensor>,
    ) -> Result<HookResult> {
        let stats = self.process_batch(
            module,
            losses,
            &data.observations,
            self.algorithm.reporter.is_some(),
        )?;
        self.record_batch(stats);
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<T: TrajectoryBatch<Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandleLearner<P>,
        batches: &[T],
    ) -> Result<HookResult> {
        if self.algorithm.reporter.is_some() {
            self.report(
                batches,
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
    fn before_learning_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 1>>>(
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

    fn rollout_hook<T: TrajectoryBatch<burn::Tensor<B::InnerBackend, 1>>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnLearner<B, P>,
        batches: &[T],
    ) -> Result<HookResult> {
        if !self.algorithm.finish_epoch() {
            return Ok(HookResult::Continue);
        }
        if self.algorithm.reporter.is_some() {
            self.report(
                batches,
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
        data: &PPOBatchData<burn::Tensor<B, 1>>,
    ) -> Result<HookResult> {
        let stats = self.process_batch(
            module,
            losses,
            &data.observations,
            self.algorithm.reporter.is_some(),
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
        self.record_batch(stats, clip_fraction, approx_kl);
        Ok(self.algorithm.check_kl(approx_kl))
    }
}

impl<P: CandlePolicy> PPOHook<CandleLearner<P>> for LearningHook<CandleLearner<P>, PPOSettings> {
    fn before_learning_hook<T: TrajectoryBatch<Tensor>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut CandleLearner<P>,
        _batches: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        let progress = self.prepare(module, advantages);
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
        if self.algorithm.reporter.is_some() {
            self.report(
                batches,
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
            self.algorithm.reporter.is_some(),
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
        self.record_batch(stats, clip_fraction, approx_kl);
        Ok(self.algorithm.check_kl(approx_kl))
    }
}
