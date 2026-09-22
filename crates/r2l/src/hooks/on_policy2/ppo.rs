use std::{marker::PhantomData, sync::mpsc::Sender};

use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    ppo::{PPOBatchData, PPOHook, PPOParams},
};
use r2l_burn::learning_module::{
    BurnPolicy, PolicyValueLearner as BurnPolicyValueLearner,
    PolicyValueLosses as BurnPolicyValueLosses,
};
use r2l_candle::learning_module::{
    CandlePolicy, PolicyValueLearner as CandlePolicyValueLearner,
    PolicyValueLosses as CandlePolicyValueLosses,
};
use r2l_core::{
    HookResult,
    buffers::TrajectoryBatch,
    error::{Error, ResourceInterrupted, Result},
    on_policy::learning_module::OnPolicyLearner,
    tensor::R2lTensor,
};

use super::{LearningRateScheduler, coordinator::SharedCoordinator};
use crate::hooks::ppo::{ClipRangeSchedule, PPOMinibatchStats, PPORolloutStats};

pub(crate) struct TargetKl {
    pub target: f32,
    pub target_exceeded: bool,
}

impl TargetKl {
    pub(crate) fn target_kl_exceeded(&mut self) -> bool {
        std::mem::take(&mut self.target_exceeded)
    }
}

pub(crate) struct PPORolloutReporter {
    report: PPORolloutStats,
    tx: Option<Sender<PPORolloutStats>>,
    log_progress: bool,
    unfinished_episode_rewards: Vec<f32>,
    latest_average_reward: f32,
}

impl PPORolloutReporter {
    /// Creates a reporter when logging or channel delivery is enabled.
    ///
    /// # Arguments
    ///
    /// * `tx` - Optional channel receiving each rollout's statistics.
    /// * `log_progress` - Whether to print rollout statistics.
    /// * `n_envs` - Number of environment reward streams to track.
    pub(crate) fn new(
        tx: Option<Sender<PPORolloutStats>>,
        log_progress: bool,
        n_envs: usize,
    ) -> Option<Self> {
        if tx.is_some() || log_progress {
            Some(Self {
                report: PPORolloutStats::default(),
                tx,
                log_progress,
                unfinished_episode_rewards: vec![0.; n_envs],
                latest_average_reward: 0.,
            })
        } else {
            None
        }
    }

    fn update_average_reward<T: r2l_core::tensor::R2lTensor, B: TrajectoryBatch<T>>(
        &mut self,
        batches: &[B],
    ) {
        let mut completed_episode_rewards = vec![];
        for (running_reward, batch) in self
            .unfinished_episode_rewards
            .iter_mut()
            .zip(batches.iter())
        {
            for (reward, done) in batch.rewards().iter().copied().zip(
                batch
                    .terminated()
                    .iter()
                    .zip(batch.truncated().iter())
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
        self.report.average_reward = self.latest_average_reward;
    }

    fn send_report(&mut self, rollout_idx: usize, total_rollouts: Option<usize>) -> Result<()> {
        self.report.rollout_idx = rollout_idx;
        self.report.total_rollouts = total_rollouts;
        let progress = std::mem::take(&mut self.report);
        if self.log_progress {
            println!("{progress}");
        }
        if let Some(tx) = &self.tx {
            tx.send(progress).map_err(|error| {
                Error::ResourceInterrupted(ResourceInterrupted {
                    resource: "PPO rollout reporter".into(),
                    details: error.to_string(),
                })
            })?;
        }
        self.report.average_reward = self.latest_average_reward;
        Ok(())
    }
}

/// Learning behavior for PPO optimization.
///
/// Applies learning-rate and clip-range schedules using the same collection progress.
///
/// This hook applies the crate's standard PPO training behavior: advantage
/// normalization when enabled, repeated PPO epochs, optional value-loss
/// weighting, optional entropy regularization, optional gradient clipping,
/// optional target-KL early stopping, and optional rollout reporting through
/// [`PPORolloutStats`].
///
/// The generic parameter tracks the concrete learner backend and is not
/// usually named directly by callers.
pub(crate) struct PPOLearningHook<T = ()> {
    pub(crate) normalize_advantage: bool,
    pub(crate) total_epochs: usize,
    pub(crate) entropy_coeff: f32,
    pub(crate) vf_coeff: Option<f32>,
    pub(crate) target_kl: Option<TargetKl>,
    pub(crate) gradient_clipping: Option<f32>,
    pub(crate) current_epoch: usize,
    pub(crate) reporter: Option<PPORolloutReporter>,
    pub(crate) coordinator: SharedCoordinator,
    pub(crate) learning_rate_scheduler: LearningRateScheduler,
    pub(crate) clip_range_schedule: ClipRangeSchedule,
    pub(crate) _lm: PhantomData<T>,
}

impl<T> PPOLearningHook<T> {
    fn update_schedules<M: OnPolicyLearner>(&self, params: &mut PPOParams, module: &mut M) {
        let progress_remaining = self.coordinator.borrow().progress_remaining();
        self.learning_rate_scheduler
            .update(progress_remaining, module);
        params.clip_range = match self.clip_range_schedule {
            ClipRangeSchedule::Constant(clip_range) => clip_range,
            ClipRangeSchedule::Linear(initial_clip_range) => {
                initial_clip_range * progress_remaining as f32
            }
        };
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PPOHook<BurnPolicyValueLearner<B, D>>
    for PPOLearningHook<BurnPolicyValueLearner<B, D>>
{
    fn before_learning_hook<
        BT: TrajectoryBatch<burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    >(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnPolicyValueLearner<B, D>,
        _batches: &[BT],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        self.current_epoch = 0;
        self.update_schedules(params, module);
        if self.normalize_advantage {
            advantages.normalize();
        }
        if let Some(max_grad_norm) = self.gradient_clipping {
            module.set_grad_clipping(GradientClipping::Norm(max_grad_norm));
        }
        Ok(HookResult::Continue)
    }

    fn rollout_hook<BT: TrajectoryBatch<burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnPolicyValueLearner<B, D>,
        batches: &[BT],
    ) -> Result<HookResult> {
        self.current_epoch += 1;
        let target_kl_exceeded = if let Some(target_kl) = &mut self.target_kl {
            target_kl.target_kl_exceeded()
        } else {
            false
        };
        let should_stop = self.current_epoch == self.total_epochs || target_kl_exceeded;
        if should_stop {
            if let Some(reporter) = &mut self.reporter {
                reporter.update_average_reward(batches);
                reporter.report.std = module.policy().std()?;
                reporter.report.learning_rate = module.policy_learning_rate();
                reporter.report.clip_range = params.clip_range;
                let (rollout_idx, total_rollouts) = {
                    let coordinator = self.coordinator.borrow();
                    (
                        coordinator.completed_rollouts() + 1,
                        coordinator.total_rollouts(),
                    )
                };
                reporter.send_report(rollout_idx, total_rollouts)?;
            }
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnPolicyValueLearner<B, D>,
        losses: &mut BurnPolicyValueLosses<B>,
        data: &PPOBatchData<burn::Tensor<B, 1>>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let entropy_loss = entropy.neg() * self.entropy_coeff;
        let approx_kl = {
            let ratio = data.ratio.to_vec()?;
            let log_ratio = data.logp_diff.to_vec()?;
            ratio
                .iter()
                .zip(log_ratio.iter())
                .map(|(ratio, log_ratio)| (ratio - 1.) - log_ratio)
                .sum::<f32>()
                / ratio.len() as f32
        };

        if let Some(PPORolloutReporter { report, .. }) = &mut self.reporter {
            let ratio = data.ratio.to_vec()?;
            let clip_fraction = ratio
                .iter()
                .filter(|value| (**value - 1.).abs() > params.clip_range)
                .count() as f32
                / ratio.len() as f32;
            report.minibatch_stats.push(PPOMinibatchStats {
                clip_fraction,
                policy_loss: losses.policy_loss.to_vec()?[0],
                entropy_loss: entropy_loss.to_vec()?[0],
                value_loss: losses.value_loss.to_vec()?[0],
                approx_kl,
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss);
        }
        if let Some(target_kl) = &mut self.target_kl {
            if approx_kl > 1.5 * target_kl.target {
                target_kl.target_exceeded = true;
                Ok(HookResult::Break)
            } else {
                Ok(HookResult::Continue)
            }
        } else {
            Ok(HookResult::Continue)
        }
    }
}

impl<P: CandlePolicy> PPOHook<CandlePolicyValueLearner<P>>
    for PPOLearningHook<CandlePolicyValueLearner<P>>
{
    fn before_learning_hook<BT: TrajectoryBatch<candle_core::Tensor>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut CandlePolicyValueLearner<P>,
        _batches: &[BT],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        self.current_epoch = 0;
        self.update_schedules(params, module);
        if self.normalize_advantage {
            advantages.normalize();
        }
        module.set_grad_clipping(self.gradient_clipping);
        Ok(HookResult::Continue)
    }

    fn rollout_hook<BT: TrajectoryBatch<candle_core::Tensor>>(
        &mut self,
        params: &mut PPOParams,
        module: &mut CandlePolicyValueLearner<P>,
        batches: &[BT],
    ) -> Result<HookResult> {
        self.current_epoch += 1;
        let target_kl_exceeded = if let Some(target_kl) = &mut self.target_kl {
            target_kl.target_kl_exceeded()
        } else {
            false
        };
        let should_stop = self.current_epoch == self.total_epochs || target_kl_exceeded;
        if should_stop {
            if let Some(reporter) = &mut self.reporter {
                reporter.update_average_reward(batches);
                reporter.report.std = module.policy().std()?;
                reporter.report.learning_rate = module.policy_learning_rate();
                reporter.report.clip_range = params.clip_range;
                let (rollout_idx, total_rollouts) = {
                    let coordinator = self.coordinator.borrow();
                    (
                        coordinator.completed_rollouts() + 1,
                        coordinator.total_rollouts(),
                    )
                };
                reporter.send_report(rollout_idx, total_rollouts)?;
            }
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        params: &mut PPOParams,
        module: &mut CandlePolicyValueLearner<P>,
        losses: &mut CandlePolicyValueLosses,
        data: &PPOBatchData<candle_core::Tensor>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
        let ratio = data.ratio.detach();
        let log_ratio = data.logp_diff.detach();
        let approx_kl = ratio
            .sub(&candle_core::Tensor::ones_like(&ratio)?)?
            .sub(&log_ratio)?
            .mean_all()?
            .to_scalar::<f32>()?;
        if let Some(PPORolloutReporter { report, .. }) = &mut self.reporter {
            let clip_fraction = (&data.ratio - 1.)?
                .abs()?
                .gt(params.clip_range)?
                .to_dtype(candle_core::DType::F32)?
                .mean_all()?
                .to_scalar::<f32>()?;
            report.minibatch_stats.push(PPOMinibatchStats {
                clip_fraction,
                policy_loss: losses.policy_loss.to_scalar()?,
                entropy_loss: entropy_loss.to_scalar()?,
                value_loss: losses.value_loss.to_scalar()?,
                approx_kl,
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(&entropy_loss)?;
        }
        if let Some(target_kl) = &mut self.target_kl {
            if approx_kl > 1.5 * target_kl.target {
                target_kl.target_exceeded = true;
                Ok(HookResult::Break)
            } else {
                Ok(HookResult::Continue)
            }
        } else {
            Ok(HookResult::Continue)
        }
    }
}
